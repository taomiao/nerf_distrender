import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./log", help="where to store ckpts and logs")
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--add_timestamp", type=int, default=0)
    parser.add_argument("--dataroot", type=str, default="/mnt/petrelfs/share_data/xulinning/sh_city")
    parser.add_argument("--datadir", type=str, default="3")

    parser.add_argument(
        "--progress_refresh_rate",
        type=int,
        default=10,
        help="how many iterations to show psnrs or iters",
    )

    parser.add_argument("--resMode", type=int, action="append")
    parser.add_argument("--camera", type=str, default="normal")

    # nerf branch
    parser.add_argument("--nerf_D", type=int, default=6)
    parser.add_argument("--nerf_D_a", type=int, default=2)
    parser.add_argument("--nerf_W", type=int, default=128)
    parser.add_argument("--nerf_freq", type=int, default=16)
    parser.add_argument("--n_level", type=int, default=2)
    parser.add_argument("--residnerf", type=int, default=0)

    # rendering
    parser.add_argument("--render_raw", type=int, default=0)
    parser.add_argument("--render_px", type=int, default=720)
    parser.add_argument("--render_nframes", type=int, default=100)
    parser.add_argument("--render_fps", type=int, default=30)
    parser.add_argument("--render_spherical", type=int, default=0)
    parser.add_argument("--render_skip", type=int, default=1)
    parser.add_argument("--render_pathid", type=int, default=0)

    parser.add_argument("--render_ncircle", type=float, default=1)
    parser.add_argument("--render_fov", type=float, default=65.0)
    parser.add_argument("--render_downward", type=float, default=-45.0)
    parser.add_argument("--render_spherical_zdiff", type=float, default=1.0)
    parser.add_argument("--render_spherical_radius", type=float, default=4.0)

    parser.add_argument("--train_near_far", type=float, action="append")
    parser.add_argument("--render_near_far", type=float, action="append")
    parser.add_argument("--lb", type=float, action="append")
    parser.add_argument("--ub", type=float, action="append")
    parser.add_argument("--render_lb", type=float, action="append")
    parser.add_argument("--render_ub", type=float, action="append")
    parser.add_argument("--subfolder", type=str, action="append")

    parser.add_argument("--filter_ray", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--distort", action="store_true")
    # parser.add_argument('--charbonier', action='store_true')

    # training schedule
    parser.add_argument("--add_upsample", type=int, default=-1)
    parser.add_argument("--add_lpips", type=int, default=-1)
    # parser.add_argument('--cal_lpips_every', type=int, default=2000)
    parser.add_argument("--add_nerf", type=int, default=-1)
    parser.add_argument("--add_distort", type=int, default=-1)
    ####
    parser.add_argument("--ndims", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=128)

    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--nonlinear_density", action="store_true")

    parser.add_argument("--run_nerf", type=int, default=0)
    parser.add_argument("--n_importance", type=int, default=128)
    parser.add_argument("--nerf_n_importance", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--downsample_train", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="TensorVMSplit")

    # loader options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--render_batch_size", type=int, default=8192)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument("--dataset_name", type=str, default="blender")
    parser.add_argument("--use_preprocessed_data", type=int, default=0)
    parser.add_argument("--processed_data_type", type=str, default="ceph")
    parser.add_argument("--processed_data_folder", type=str, default="s3://preprocessed_data/ds10_leftsmall/")

    # training options
    # random seed
    parser.add_argument("--random_seed", type=int, default=20211202, help="random seed")

    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02, help="learning rate")
    parser.add_argument("--lr_basis", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=-1,
        help="number of iterations the lr will decay to the target ratio; -1 will set it to n_iters",
    )
    parser.add_argument(
        "--lr_decay_target_ratio",
        type=float,
        default=0.1,
        help="the target decay ratio; after decay_iters inital lr decays to lr*ratio",
    )
    parser.add_argument(
        "--lr_upsample_reset",
        type=int,
        default=1,
        help="reset lr to inital after upsampling",
    )

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0, help="loss weight")
    parser.add_argument("--L1_weight_rest", type=float, default=0, help="loss weight")
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help="loss weight")
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help="loss weight")
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help="loss weight")

    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument(
        "--rm_weight_mask_thre",
        type=float,
        default=0.0001,
        help="mask points in ray marching",
    )
    parser.add_argument(
        "--alpha_mask_thre",
        type=float,
        default=0.0001,
        help="threshold for creating alpha mask volume",
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=25,
        help="scaling sampling distance for computation",
    )
    parser.add_argument(
        "--density_shift",
        type=float,
        default=-10,
        help="shift density in softplus; making density = 0  when feature == 0",
    )

    # network decoder
    parser.add_argument("--density_mlp", default=False, action="store_true")
    parser.add_argument("--shadingMode", type=str, default="MLP_Fea", help="which shading mode to use")
    parser.add_argument("--pos_pe", type=int, default=6, help="number of pe for pos")
    parser.add_argument("--view_pe", type=int, default=6, help="number of pe for view")
    parser.add_argument("--fea_pe", type=int, default=6, help="number of pe for features")
    parser.add_argument("--featureC", type=int, default=128, help="hidden feature channel in MLP")
    parser.add_argument("--bias_enable", type=int, default=0, help="control the bias of app MLP")
    parser.add_argument("--encode_app", default=False, action="store_true")

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=1)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--render_path2", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument(
        "--lindisp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default="softplus")
    parser.add_argument("--ndc_ray", type=int, default=0)
    parser.add_argument(
        "--nSamples",
        type=int,
        default=1e6,
        help="sample point each ray, pass 1e6 if automatic adjust",
    )
    parser.add_argument("--step_ratio", type=float, default=0.5)
    parser.add_argument(
        "--compute_extra_metrics",
        type=int,
        default=1,
        help="whether to compute lpips metric",
    )

    # blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )

    parser.add_argument("--N_voxel_init", type=int, default=100**3)
    parser.add_argument("--N_voxel_final", type=int, default=300**3)
    parser.add_argument("--alpha_grid_reso", type=int, default=256**3)
    parser.add_argument("--progressive_alpha", type=int, default=0)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument("--idx_view", type=int, default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5, help="N images to vis")
    parser.add_argument("--vis_every", type=int, default=10000, help="frequency of visualize the image")

    # profiler tool
    parser.add_argument("--train_profile", type=int, default=0)
    parser.add_argument("--render_profile", type=int, default=0)
    parser.add_argument(
        "--prof_dir",
        type=str,
        default="./prof_dir/",
        help="where to store profile files",
    )
    parser.add_argument("--profile_rank0_only", type=int, default=1)
    parser.add_argument("--op_prof", type=int, default=1)
    parser.add_argument("--profile_flush", type=int, default=1)

    # sanity tool
    parser.add_argument("--train_sanity", type=int, default=0)
    parser.add_argument("--render_sanity", type=int, default=0)
    parser.add_argument("--baseline_dir", type=str, default=None, help="where to store profile files")

    # parallel mode
    parser.add_argument("--DDP", type=int, default=0)
    parser.add_argument("--channel_parallel", type=int, default=0)
    parser.add_argument("--plane_parallel", type=int, default=0)
    parser.add_argument("--block_parallel", type=int, default=0)
    parser.add_argument("--model_parallel_and_DDP", type=int, default=0)
    # plane division
    parser.add_argument(
        "--plane_division",
        type=int,
        action="append",
        help="plane_division setting when using plane_parallel or block_parallel",
    )
    # block parallel ckpt type, used in rendering
    parser.add_argument(
        "--ckpt_type",
        type=str,
        default="full",
        help=(
            "loaded checkpoint type of block parallel when rendering.\nfull: ckpt is fully merged (default); part: ckpt"
            " is a part of fully-merged ckpt; sub: ckpt is not merged."
        ),
    )
    # full grid density plane & line computed offline, used in fullgrid rendering
    parser.add_argument(
        "--fullgridnpz",
        type=str,
        default=None,
        help=(
            "loading the fullgrid density plane & line when rendering in fullgrid mode.\n"
            "It is now computed offline, which also can be done online."
        ),
    )
    # render mode
    parser.add_argument("--DDP_render", type=int, default=0, help="Whether to use ddp rendering. False by default.")

    # ci mode
    parser.add_argument("--ci_test", type=int, default=0)

    # jit and kernel opt
    parser.add_argument("--render_kernel_opt", default=False, action="store_true")
    parser.add_argument("--render_jit_opt", default=False, action="store_true")

    # all block render test
    parser.add_argument("--all_block_render_test", default=False, action="store_true")
    parser.add_argument("--test_all", default=False, action="store_true")
    parser.add_argument("--e2e_testing", default=False, action="store_true")

    # aliyun
    parser.add_argument("--aliyun", default=True, action="store_true")

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()


def check_args(args):
    assert (
        sum([args.plane_parallel, args.channel_parallel, args.block_parallel]) <= 1
    ), "Only one of the channel/plane/block parallel modes can be True"
    plane_division = args.plane_division
    if args.model_parallel_and_DDP:
        assert args.use_preprocessed_data
        assert args.block_parallel or args.channel_parallel or args.plane_parallel
        if args.plane_parallel or args.block_parallel:
            assert args.world_size % (plane_division[0] * plane_division[1]) == 0
    else:
        # plane parallel
        if args.plane_parallel or args.block_parallel:
            assert (
                plane_division[0] * plane_division[1] == args.world_size
            ), "world size is not equal to num of divided planes"

    if args.use_preprocessed_data:
        assert args.add_lpips == -1
        assert args.batch_size % 8192 == 0
