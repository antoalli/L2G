import os.path as osp
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
from dataset.sample_grasp_dataset import *
from l2g_core.graspsamplenet import GraspSampleNet
from l2g_core.utils.grasp_utils import cal_accuracy
from utils_alli import *
import wandb
import time


def adjust_learning_rate(optimizer, epoch, lr, rate=0.5, step=100):
    """
    lr scheduling function from GPNet
    @param optimizer: the optimizer to adjust the lr for
    @param epoch: the current epoch
    @param lr: the INITIAL lr
    @param rate: the lr multiplying factor
    @param step: after how many epochs to re-schedule the lr
    @return: the lr to use for the current epoch
    """
    lr = lr * (rate ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=str, default="l2g_experiments",
                        help="directory for experiments logging")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
    parser.add_argument('--workers', type=int, default=12, help='the num of worker for the dataloader')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--lr_step', type=int, default=999, help='learning rate scheduler step')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--epochs', type=int, default=400, help='num epochs to train')
    parser.add_argument('--num_warmup_it', type=int, default=0,
                        help='num warmup iterations (in reality epochs) for contact points sampling part')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to restart training from')
    parser.add_argument('--max_points', type=int, default=20000, help="skip pointclouds with more #points > max_points")
    parser.add_argument('--save_it', type=int, default=10)
    parser.add_argument('--project', type=str, default='grasping_l2g_new_loss_pd_fix', help="wandb project")
    parser.add_argument('--data_root', type=str, default="/data/datasets/GPNet_release_data", help="path to dataset")

    # grasp data params
    parser.add_argument('--view', type=int, default=0, help='test time view')
    parser.add_argument('--grasp_sample_num', type=int, default=1000,
                        help="number of grasp to randomly sample from the dataset")
    parser.add_argument('--grasp_positive_ratio', type=float, default=0.3,
                        help="ratio of positive annotated grasps")
    parser.add_argument('--contact_th', type=float, default=0.0035,
                        help='threshold to determine if a contact point is close enough to the pc')
    parser.add_argument("--use_angle_feat", type=str2bool, nargs='?', const=True, default=True,
                        help="feature input to grasp clf. default is True")
    parser.add_argument('--use_all_grasp_info', action='store_true')
    parser.add_argument('--lamb', default=0.1, type=float, help='lambda for multi-angle loss.')
    ################################################################################################
    # local neighborhood at first (sampled) contact point
    # TODO: this should be a Ball Query rather than knn search!!!
    parser.add_argument('--neigh_size', type=int, default=100,
                        help='size neighborhood for feature aggr at sampled first contact')
    ################################################################################################
    parser.add_argument('--neigh_aggr', type=str, default='w_avg', choices=['avg', 'max', 'w_avg'],
                        help='feat aggregation type at sampled first contact')

    # sampling model params
    parser.add_argument('--feat', type=str, default='pointnet1', choices=['pointnet1', 'pointnet2', 'deco'],
                        help="feature extractor to use")
    parser.add_argument('--deco_config', type=str, default='./deco_lib/deco_franklin.yaml',
                        help='deco configuration file - pretext loaded or not depending on the Deco config')
    parser.add_argument('--matching_policy', type=str, default='soft', choices=['hard', 'soft'],
                        help='whether to get only the closest point in the pc or all those within a certain threshold')
    parser.add_argument('--sampled_grasps', type=int, default=400, help='number of grasps to generate')
    parser.add_argument('--sample_group_size', type=int, default=10, help='size neighborhood for the projection phase')
    parser.add_argument('--alpha', type=float, default=10.0, help='simplification loss (sampling) weighting factor')
    parser.add_argument("--train_temperature", type=str2bool, nargs='?', const=True, default=True,
                        help="sampling: wheter to train or not the temperature parameter")
    parser.add_argument('--train_separately', action='store_true',
                        help='train the sampler and the grasp heads almost separately (feature extractor in common)')

    return parser.parse_args()


def parse_experiment():
    args = parse_arguments()

    # set experiment seed
    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    args.optimizer = str(args.optimizer).lower()  # lower optimizer name before use
    exp_name = f"{args.feat}" \
               f"_neigh-size{args.neigh_size}" \
               f"_use-angle-feat_{args.use_angle_feat}" \
               f"_lambda{args.lamb}" \
               f"_samp-grasp{args.sampled_grasps}" \
               f"_match-policy-{args.matching_policy}" \
               f"_posi-ratio-{args.grasp_positive_ratio}" \
               f"_warmup_it_{args.num_warmup_it}" \
               f"/" \
               f"opt-{args.optimizer}" \
               f"_lr{args.lr}" \
               f"_lr-step{args.lr_step}" \
               f"_wd{args.wd}" \
               f"_epochs{args.epochs}" \
               f"_seed{args.seed}"

    if args.use_all_grasp_info:
        exp_name += '_all-grasp-info'
    if args.train_separately:
        exp_name += '_train-sep'

    args.exp_dir = osp.join(args.checkpoints_dir, exp_name)
    args.models_dir = osp.join(args.exp_dir, "checkpoints")
    safe_make_dirs([args.models_dir])
    io_logger = IOStream(osp.join(args.exp_dir, "log.txt"))
    err_logger = IOStream(osp.join(args.exp_dir, "errors.txt"))
    shutil.copy(osp.basename(__file__), osp.join(args.exp_dir, osp.basename(__file__)))

    # using wandb instead of tensorboard logger
    # tb_writer = SummaryWriter(logdir=args.exp_dir)
    wandb.login()
    wandb.init(
        project=args.project,
        name=exp_name,
        config={'arguments': args},
        entity="shannon94")

    return args, io_logger, err_logger


def warmup_sampler(warmup_it, model, dataloader, optimizer, opt):
    """
    Training only the contact points sampling part for 'warmup_it' EPOCHS

    Parameters
    ----------
    warmup_it
    model
    dataloader
    optimizer
    opt

    Returns
    -------

    """
    warmup_res = {
        "warmup_sampling_gen_pc": 0,
        "warmup_sampling_max_gen_pc": 0,
        "warmup_sampling_pc_gen": 0,
        "warmup_sampling_simplification": 0,
        "warmup_sampling_projection": 0,
        "warmup_sampling_tot": 0,
        "warmup_tot_loss": 0,
    }

    # MODEL: train mode
    model.train()

    for i, batch_data in enumerate(tqdm(dataloader, desc=f"warmup %d" % warmup_it)):
        pc, first_contact_pc_indexes, contacts, angles, scores, contact_indexes, grasp_indexes, shape = batch_data

        # skip shapes to avoid OOM
        if pc.shape[1] > opt.max_points:
            continue

        # deleting the batch size dimension (bs = 1)
        first_contact_pc_indexes = first_contact_pc_indexes.squeeze(0).long()
        contact_indexes = contact_indexes.squeeze(0).long()
        grasp_indexes = grasp_indexes.squeeze(0).long()

        # cannot perform the sampling operation if there is no sampling truth
        if len(first_contact_pc_indexes) == 0 or len(contact_indexes) == 0:
            continue

        # extracting contact points related to positive annotated grasp
        # since these are the ones we want to learn to sample
        first_contacts_pc = pc[:, first_contact_pc_indexes, :]  # [B x G x 3]

        # training data: grasp is parametrized in such a way (c1, c2, theta)
        # c1: first contact point
        # c2: second contact point, do not belong to the partial view
        # theta: the corresponding angle
        first_contacts = contacts[:, contact_indexes[:, 0], contact_indexes[:, 1], :]  # [B x G x 3]
        second_contacts = contacts[:, contact_indexes[:, 0], contact_indexes[:, 2], :]  # [B x G x 3]
        angles = angles[:, contact_indexes[:, 0]].unsqueeze(-1)  # [B x G x 1]
        scores = scores[:, contact_indexes[:, 0]].unsqueeze(-1)  # [B x G x 1]

        gt_sampling = first_contacts_pc
        all_gt_grasps = torch.cat((first_contacts, second_contacts, angles), dim=-1)  # [B x G x 7]
        # all_gt_grasp_scores = scores
        # all_gt_positive_grasps = all_gt_grasps[:, torch.nonzero(scores, as_tuple=True)[1], :]

        # grasp indexes are computed such that the grasp classifier gets a specific number of input grasps
        # with a specific balancing of the labels
        gt_grasps = all_gt_grasps[:, grasp_indexes]
        # gt_grasp_scores = all_gt_grasp_scores[:, grasp_indexes]

        # model forward
        pc = pc.float().cuda()  # need to convert to float to be consistent with the pointnet2 implementation
        gt_sampling = gt_sampling.float().cuda()
        gt_grasps = gt_grasps.float().cuda()
        sampling_output = model(pc, gt_grasps, gt_sampling)[0]

        """
        Sampling loss is made of two components
            1 - simplification_loss : chamfer distance between set of generated and sampled points
            2 - projection_loss : learned temperature
        """
        generated, _ = sampling_output
        gen_pc_loss, max_gen_pc_loss, pc_gen_loss, simplification_loss = \
            model.sampler.get_simplification_loss(generated, gt_sampling)
        projection_loss = model.sampler.get_projection_loss()
        sampling_loss = opt.alpha * simplification_loss + projection_loss

        optimizer.zero_grad()
        sampling_loss.backward()
        optimizer.step()

        # logging
        # losses
        warmup_res["warmup_sampling_simplification"] += simplification_loss.item()
        warmup_res["warmup_sampling_projection"] += projection_loss.item()
        warmup_res["warmup_sampling_tot"] += sampling_loss.item()

    for k in warmup_res.keys():
        # len(dataloader) since batch_size=1
        warmup_res[k] = warmup_res[k] / len(dataloader)

    # log also current epoch learning rate
    warmup_res["warmup_sampling_lr"] = optimizer.param_groups[0]['lr']
    return warmup_res


def train_one_epoch(epoch, glob_it, model, dataloader, optimizer, err_logger, opt):
    train_res = {
        "sampling_gen_pc": 0,
        "sampling_max_gen_pc": 0,
        "sampling_pc_gen": 0,
        "sampling_simplification": 0,
        "sampling_projection": 0,
        "sampling_tot": 0,
        "grasp_prediction_center": 0,
        "grasp_prediction_angle": 0,
        "grasp_prediction_tot": 0,
        "grasp_classification": 0,
        "tot_loss": 0,
        "grasp_clf_accuracy": 0,
        "grasp_clf_recall": 0
    }

    # MODEL: train mode
    model.train()

    # for i, batch_data in enumerate(tqdm(dataloader, desc=f"train ep {epoch}")):
    for i, batch_data in enumerate(dataloader, 0):
        glob_it += 1  # update global iteration counter
        pc, first_contact_pc_indexes, contacts, angles, scores, contact_indexes, grasp_indexes, shape = batch_data

        # skip shapes to avoid OOM
        if pc.shape[1] > opt.max_points:
            err_logger.cprint(f"Epoch {epoch} - skipped shape {shape} because has {pc.shape[1]} points")
            continue

        # deleting the batch size dimension (bs = 1)
        first_contact_pc_indexes = first_contact_pc_indexes.squeeze(0).long()
        contact_indexes = contact_indexes.squeeze(0).long()
        grasp_indexes = grasp_indexes.squeeze(0).long()

        # cannot perform the sampling operation if there is no sampling truth
        if len(first_contact_pc_indexes) == 0 or len(contact_indexes) == 0:
            err_logger.cprint(f"Epoch {epoch} - unable to compute truth for shape {shape}")
            continue

        # extracting contact points related to positive annotated grasp
        # since these are the ones we want to learn to sample
        first_contacts_pc = pc[:, first_contact_pc_indexes, :]  # [B x G x 3]

        # training data: grasp is parametrized in such a way (c1, c2, theta)
        # c1: first contact point
        # c2: second contact point, do not belong to the partial view
        # theta: the corresponding angle
        first_contacts = contacts[:, contact_indexes[:, 0], contact_indexes[:, 1], :]  # [B x G x 3]
        second_contacts = contacts[:, contact_indexes[:, 0], contact_indexes[:, 2], :]  # [B x G x 3]
        angles = angles[:, contact_indexes[:, 0]].unsqueeze(-1)  # [B x G x 1]
        scores = scores[:, contact_indexes[:, 0]].unsqueeze(-1)  # [B x G x 1]

        gt_sampling = first_contacts_pc
        all_gt_grasps = torch.cat((first_contacts, second_contacts, angles), dim=-1)  # [B x G x 7]
        all_gt_grasp_scores = scores
        all_gt_positive_grasps = all_gt_grasps[:, torch.nonzero(scores, as_tuple=True)[1], :]

        # grasp indexes are computed such that the grasp classifier gets a specific number of input grasps
        # with a specific balancing of the labels
        gt_grasps = all_gt_grasps[:, grasp_indexes]
        gt_grasp_scores = all_gt_grasp_scores[:, grasp_indexes]

        # model forward
        pc = pc.float().cuda()  # need to convert to float to be consistent with the pointnet2 implementation
        gt_sampling = gt_sampling.float().cuda()
        gt_grasps = gt_grasps.float().cuda()
        gt_positive_grasps = all_gt_positive_grasps.float().cuda()
        gt_grasp_scores = gt_grasp_scores.float().cuda()
        sampling_output, generated_grasps, predicted_grasps_scores = model(pc, gt_grasps, gt_sampling)

        """
        Sampling loss is made of two components
            1 - simplification_loss : chamfer distance between set of generated and sampled points
            2 - projection_loss : learned temperature
        """

        generated, _ = sampling_output
        gen_pc_loss, max_gen_pc_loss, pc_gen_loss, simplification_loss = \
            model.sampler.get_simplification_loss(generated, gt_sampling)
        projection_loss = model.sampler.get_projection_loss()
        sampling_loss = opt.alpha * simplification_loss + projection_loss

        # GRASP PREDICTION LOSS
        # set as truth only the positive grasps
        center_loss, angle_loss, grasp_prediction_loss = model.grasp_predictor.get_prediction_loss(
            generated_grasps, gt_positive_grasps, angle_contribution=opt.lamb)

        # GRASP CLASSIFICATION LOSS
        grasp_classification_loss = model.grasp_classifier.get_classification_loss(
            predicted_grasps_scores, gt_grasp_scores)

        # print("sampling_loss: ", sampling_loss, sampling_loss.shape)
        # print("grasp_prediction_loss: ", grasp_prediction_loss, grasp_prediction_loss.shape)
        # print("grasp_classification_loss: ", grasp_classification_loss, grasp_classification_loss.shape)
        loss = sampling_loss + grasp_prediction_loss + grasp_classification_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # grasp clf accuracy
        grasp_clf_accuracy, grasp_clf_recall = cal_accuracy(
            gt_grasp_scores.view(-1), predicted_grasps_scores.view(-1), recall=True)

        # logging
        # losses => must go down
        train_res["sampling_gen_pc"] += gen_pc_loss.item()
        train_res["sampling_max_gen_pc"] += max_gen_pc_loss.item()
        train_res["sampling_pc_gen"] += pc_gen_loss.item()
        train_res["sampling_simplification"] += simplification_loss.item()
        train_res["sampling_projection"] += projection_loss.item()
        train_res["sampling_tot"] += sampling_loss.item()
        train_res["grasp_prediction_center"] += center_loss.item()
        train_res["grasp_prediction_angle"] += angle_loss.item()
        train_res["grasp_prediction_tot"] += grasp_prediction_loss.item()
        train_res["grasp_classification"] += grasp_classification_loss.item()
        train_res["tot_loss"] += loss.item()
        # accuracies => must increase
        # grasp clf acc + recall
        train_res["grasp_clf_accuracy"] += grasp_clf_accuracy
        train_res["grasp_clf_recall"] += grasp_clf_recall

    for k in train_res.keys():
        # len(dataloader) since batch_size=1
        train_res[k] = train_res[k] / len(dataloader)

    # log also current epoch learning rate
    train_res["lr"] = optimizer.param_groups[0]['lr']

    return train_res


def main():
    opt, io, error_logger = parse_experiment()
    io.cprint(f"Arguments: {opt} \n")

    train_dataset = SampleGraspData(
        data_root=opt.data_root,
        split="train",
        sample_num=opt.grasp_sample_num,
        positive_ratio=opt.grasp_positive_ratio,
        contact_th=opt.contact_th,
        matching_policy=opt.matching_policy,
        view=opt.view,
        fixed_view=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, drop_last=False,
        pin_memory=True)

    # MODEL DEFINITION
    model = GraspSampleNet(
        feat_extractor=opt.feat,
        deco_config_path=opt.deco_config,
        sampled_grasps=opt.sampled_grasps,
        sample_group_size=opt.sample_group_size,
        simp_loss='chamfer',
        train_temperature=opt.train_temperature,
        neigh_size=opt.neigh_size,
        use_all_grasp_info=opt.use_all_grasp_info,
        use_contact_angle_feat=opt.use_angle_feat,
        angle_feat_depth=2,
        projected_feat_aggregation=opt.neigh_aggr,
        train_separately=opt.train_separately,
        bn=False
    )

    # sanity check
    # avoid spurious BN layers in the netwoek
    for name, child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            assert False, "BatchNorm found in model!!!"

    # MOVE MODEL TO CUDA
    model = model.cuda()
    wandb.watch(
        model, criterion=None, log="gradients", log_freq=100, idx=None, log_graph=True
    )

    # BUILD OPTIMIZER
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, weight_decay=opt.wd, momentum=opt.momentum)
    else:
        raise ValueError(f"Wrong optimizer: {opt.optimizer}")

    start_epoch, glob_it = 0, 0

    if opt.resume:
        assert osp.isfile(opt.resume), "Wrong resume path"
        io.cprint(f"Resuming from {opt.resume}.")
        ckt = torch.load(opt.resume)
        start_epoch = ckt['epoch']
        glob_it = ckt['glob_it']
        # print('opt state: ', ckt['optimizer_state'])
        print("load model state: ", model.load_state_dict(ckt['model']))
        print("load optimizer state: ", optimizer.load_state_dict(ckt['optimizer_state']))
        # print("load scheduler state: ", scheduler.load_state_dict(ckt['scheduler_state']))
        del ckt

    ############
    #  WARMUP  #
    ############

    if not opt.resume:
        # warmup if not restarting from checkpoint
        for warmup_it in range(1, opt.num_warmup_it + 1):
            assert not opt.train_separately
            warmup_res = warmup_sampler(warmup_it, model, train_loader, optimizer, opt)
            wandb.log(warmup_res)
            res_str = f"Warmup it. [{warmup_it}/{opt.num_warmup_it}] - "
            for k in sorted(warmup_res.keys()):
                res_str += f"{k}: {warmup_res[k]}; "
            io.cprint(res_str + '\n')

    ##############
    #  TRAINING  #
    ##############

    # start_epoch += opt.num_warmup_it   # training starts from 'num_warmup_it'
    io.cprint(f"Training - start_epoch: {start_epoch},  glob_it: {glob_it}")

    for epoch in range(start_epoch + 1, opt.epochs + 1):
        # lr scheduling
        if opt.lr_step < opt.epochs:
            new_lr = adjust_learning_rate(optimizer=optimizer, epoch=epoch, lr=opt.lr, step=opt.lr_step)
            io.cprint(f"lr scheduling - lr: {new_lr} at epoch {epoch}")
        else:
            new_lr = opt.lr

        if not opt.train_temperature:
            # TODO: plot of learned temperature is extremely regular (???)
            model.sampler.project.update_temperature(epoch, opt.epochs)

        ep_start = time.time()
        # TRAIN ONE EPOCH
        epoch_res = train_one_epoch(epoch, glob_it, model, train_loader, optimizer, error_logger, opt)

        # pretty print epoch train results
        res_str = f"Train Epoch [{epoch}/{opt.epochs}] - "
        for k in sorted(epoch_res.keys()):
            res_str += f"{k}: {epoch_res[k]}; "

        # append epoch time
        res_str += "time: {}; \n".format(time.strftime("%M:%S", time.gmtime(time.time() - ep_start)))
        io.cprint(res_str + '\n')

        # wandb logging
        epoch_res['epoch'] = epoch
        epoch_res['scheduled_lr'] = new_lr

        wandb.log(epoch_res)

        # save checkpoint
        if epoch % opt.save_it == 0:
            torch.save({
                'args': opt,
                'epoch': epoch,
                'glob_it': glob_it,
                'model': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch_res': epoch_res
            }, osp.join(opt.models_dir, f'epoch_{epoch:03d}.pth'))


main()
sys.exit(0)
