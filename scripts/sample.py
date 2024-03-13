import argparse

from guided_diffusion import dist_util
from guided_diffusion import logger
import torch as th
import torchvision.utils as vutils
from guided_diffusion.dataloader import IRSTD_Dataset, SIRST_Dataset, NUDT_Dataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    if args.data_name == 'IRSTD':
        dataset = IRSTD_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4
    elif args.data_name == 'NUAA':
        dataset = SIRST_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4
    elif args.data_name == 'NUDT':
        dataset = NUDT_Dataset(args, args.data_dir, mode='test')
        args.in_ch = 4

    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data = iter(dataloader)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    while len(all_images) * args.batch_size < args.num_samples:
        b, m, path = next(data)  # should return an image from the dataloader "data"
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)  # add a noise channel$
        if args.data_name == 'IRSTD':
            slice_ID = path[0].split("_")[-1].split('.')[0]
        if args.data_name == 'SIRST':
            slice_ID = path[0].split('.')[0]
        if args.data_name == 'NUDT':
            slice_ID = path[0].split('.')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        model_kwargs = {}
        start.record()
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        sample, x_noisy, org, cal, cal_out = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size), img,
            step=args.diffusion_steps,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        end.record()
        th.cuda.synchronize()
        print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample

        co = th.tensor(cal_out)

        vutils.save_image(co, fp=args.out_dir + str(slice_ID) + ".png", nrow=1, padding=10)


def create_argparser():
    defaults = dict(
        data_name='IRSTD',
        data_dir=r"../data/IRSTD-1k",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path=r"../results/results_NUDT/model000000.pt",
        gpu_dev="0",
        out_dir='../sample/NUDT_000000/',
        multi_gpu=None,  # "0,1,2"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
