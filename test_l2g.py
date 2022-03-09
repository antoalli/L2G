import sys
import os
sys.path.append(os.getcwd())
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.sample_grasp_dataset import *
from l2g_core.graspsamplenet import GraspSampleNet
from l2g_core.utils.grasp_utils import reparametrize_grasps
from utils import *
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to restart training from')
    parser.add_argument('--data_root', type=str, default="/data/datasets/GPNet_release_data", help="path to dataset")
    parser.add_argument('--deco_config', type=str, default=None, help='path to DeCo configuration file')
    parser.add_argument('--views', type=str, default='0,1,2,3,4', help='views to test on')
    parser.add_argument('--save_all', action="store_true",
                        help='whether to save all the grasps or only the positively predicted ones')
    parser.add_argument('--test_type', type=str, default='test',
                        choices=['test', 'ycb8_test', 'ycb76_test'],
                        help='define the type of test to perform: '
                             'test -> ShapeNetSem8, '
                             'ycb8_test -> YCB-8 (can compute rb, cov, sim), '
                             'ycb76_test -> YCB-76 (only simulation results) ')
    return parser.parse_args()


def main():
    args = get_args()  # test arguments
    views = [int(i) for i in args.views.split(',')]
    print(f"Testing views: {views} \n\n")
    for v in views:
        print(f"Now testing view {v}")
        test(resume_path=args.resume, view=v, test_args=args)
        print("@" * 30)
        print("\n")


@torch.no_grad()
def test(resume_path, view, test_args):
    """

    Parameters
    ----------
    resume_path: path to checkpoint to resume
    view: view to perform test on
    test_args: test arguments

    Returns
    -------

    """
    ckt = torch.load(resume_path)
    test_epoch = ckt["epoch"]
    train_args = ckt["args"]  # training time arguments loaded from checkpoint
    data_root = test_args.data_root
    save_all = test_args.save_all

    # set experiment seed
    set_random_seed(train_args.seed)

    # DeCo config
    if test_args.deco_config:
        # test argument is specified
        # use case is training and testing on different machines (thus diff. paths for configuration file)
        deco_config = test_args.deco_config
    else:
        # no test argument specified
        # resume path from arguments specified at training time
        deco_config = train_args.deco_config

    test_type = test_args.test_type
    checkpoints_dir = '/'.join(resume_path.split('/')[:-1])  # folder containing all '.pth' files
    out_dir = os.path.join(checkpoints_dir, test_type, f"epoch{test_epoch}")
    print(f"Test type: {test_type}\n"
          f"Output dir: {out_dir}")

    all_grasps_dir = os.path.join(out_dir, 'view%d' % view)
    safe_make_dirs([all_grasps_dir])
    log = IOStream(osp.join(out_dir, "log.txt"))
    log_all = IOStream(os.path.join(out_dir, 'nms_poses_view%s.txt' % view))

    if test_type in ['test', 'ycb8_test']:
        test_dataset = SampleGraspData(
            data_root=data_root,
            split=test_type,
            sample_num=train_args.grasp_sample_num,
            positive_ratio=train_args.grasp_positive_ratio,
            contact_th=train_args.contact_th,
            matching_policy=train_args.matching_policy,
            view=view)
    elif test_type == 'ycb76_test':
        test_dataset = YCB76_Data(data_root=data_root, split='test', view=view)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    test_loader = DataLoader(
        test_dataset, batch_size=train_args.batch_size, num_workers=train_args.workers,
        shuffle=False, drop_last=False, pin_memory=True)

    # MODEL DEFINITION
    model = GraspSampleNet(
        feat_extractor=train_args.feat,
        deco_config_path=deco_config,
        sampled_grasps=train_args.sampled_grasps,
        sample_group_size=train_args.sample_group_size,
        simp_loss='chamfer',
        train_temperature=train_args.train_temperature,
        neigh_size=train_args.neigh_size,
        use_all_grasp_info=False,
        use_contact_angle_feat=train_args.use_angle_feat,
        angle_feat_depth=2,
        projected_feat_aggregation=train_args.neigh_aggr,
        bn=False,
        resume=True
    )

    # avoid spurious BN layers in the netwoek
    for name, child in (model.named_children()):
        if name.find('BatchNorm') != -1:
            assert False

    # move model to cuda device
    model = model.cuda()
    # load weights from ckt
    log.cprint('-' * 30)
    log.cprint(f"Resume: {resume_path} \nepoch: {test_epoch} \nview: {view}")
    res_load_weights = model.load_state_dict(ckt['model'], strict=True)
    log.cprint(f"Load weights: {res_load_weights}")
    # set eval mode
    model = model.eval()

    time_list = []  # inference time for each shape
    for _, batch_data in enumerate(tqdm(test_loader)):
        st = time.time()
        pc, shape = batch_data
        shape = shape[0]
        pc = pc.float().cuda()

        # model prediction
        (generated, matched), predicted_grasps, predicted_scores = model(pc, gt_sampling=None, gt_grasps=None)

        # log inference time
        inf_time = time.time() - st
        time_list.append(inf_time)
        print(f'pc shape: {pc.shape}, forward time: {inf_time}')

        if save_all:
            positive_idx = torch.nonzero(predicted_scores.view(-1) >= 0.).view(-1)
        else:
            positive_idx = torch.nonzero(predicted_scores.view(-1) >= 0.5).view(-1)

        if positive_idx.size(0) == 0:
            warnings.warn(f"No positive grasps for shape {shape}.")
            centers = torch.empty((0, 3))
            quaternions = torch.empty((0, 4))
            widths = torch.empty((0, 1))
            predicted_scores = torch.empty((0, 1))
        else:
            predicted_grasps = predicted_grasps[:, positive_idx, :]
            predicted_scores = predicted_scores[:, positive_idx, :]
            # getting the (center, quaternion) parametrization
            rep_predicted_grasps = reparametrize_grasps(predicted_grasps, with_width=True, gpnet_scale=True)
            centers = rep_predicted_grasps[:, :, :3].squeeze(0)
            quaternions = rep_predicted_grasps[:, :, 3:7].squeeze(0)
            widths = rep_predicted_grasps[:, :, 7].squeeze(0)
            predicted_scores = predicted_scores.view(-1)

        # pruning grasps whose center is below the ground
        centers_z = centers[:, 2]
        select = torch.nonzero((widths < 0.085) * (centers_z > 0)).view(-1)
        centers, widths, quaternions, predicted_scores = \
            centers[select], widths[select], quaternions[select], predicted_scores[select]

        centers = centers.cpu().numpy()
        quaternions = quaternions.cpu().numpy()
        widths = widths.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        sorted_idx = np.argsort(-predicted_scores)
        centers, widths, quaternions, predicted_scores = \
            centers[sorted_idx], widths[sorted_idx], quaternions[sorted_idx], predicted_scores[sorted_idx]

        all_grasps_path = os.path.join(all_grasps_dir, shape + '.npz')
        np.savez(all_grasps_path, widths=widths, centers=centers, quaternions=quaternions, scores=predicted_scores)

        log_all.cprint(shape)
        for i in range(centers.shape[0]):
            w = widths[i]
            c = centers[i]
            q = quaternions[i]
            score = predicted_scores[i]
            log_all.cprint('%f,%f,%f,%f,%f,%f,%f,%f' % (c[0], c[1], c[2], q[0], q[1], q[2], q[3], score))

    time_list = np.asarray(time_list)
    np.savetxt(fname=osp.join(out_dir, f'times_{view}.txt'), X=time_list)
    print(f"time_list: {time_list}, num shapes: {len(time_list)}")
    print(f"time_list mean: {np.mean(time_list)}")
    print(f"time_list std: {np.std(time_list)}")


if __name__ == '__main__':
    main()
    sys.exit(0)
