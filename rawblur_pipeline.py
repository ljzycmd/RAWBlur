import os
import shutil
import numpy as np
import imageio
import rawpy
import cv2
import random

from PIL import Image


def archive_frames(path):
    """
    archive the raw frame-level data into video level from raw data path: path
    """
    frames = os.listdir(path)
    frames.sort()
    for i, frame in enumerate(frames):
        if os.path.isdir(frame):
            continue
        try:
            video, frame_name = frame.split("-")
            video_path = os.path.join(path, video)
            if not os.path.exists(video_path):
                os.makedirs(video_path, exist_ok=True)
            shutil.move(os.path.join(path, frame), video_path)
            print(f"Move {frame} into {video_path} done!")
        except:
            continue


def read_raw_img(path, H=620, W=812, black_level=0):
    """
    read raw data from the .bin files
    Args:
        path: the path of the raw data
        H: the height of the raw image
        W: the width of the raw image
        black_level: the black level of the raw image
    """
    assert path, "input should not be None!"
    raw_img = np.fromfile(path, dtype="uint16")
    if len(raw_img) != H*W:  # if the file owns a header with 256 bit
        raw_img = raw_img[256:]
    raw_img = raw_img.astype(np.float32)  # convert the data type into float32 to aooid overflow
    if black_level > 0:  # black-level correction
        raw_img -= black_level
        raw_img = np.clip(raw_img, 0, 65536)  # clip the data into the range of 16bit
    # transter 12bit data into 16bit data (note that the raw data is 12bit with our data)
    raw_img = raw_img.reshape(H, W) * 2**4
    raw_img = raw_img.astype(np.uint16)
    return raw_img


def synthesize_rawblur_single_video(video_path, output_video_path=None, total_frames=33, num_sharp_frames=1, frames_used_to_synthesize=7):
    """
    synthezide a blurry video frames from high-frame rate raw sharp frames
    Note that the noise is not added in this function
    Args:
        video_path: the path of the high-frame rate raw frames of a video
        output_video_path: the path of the output blurry raw video
        total_frames: the total number of frames in the output video
        num_sharp_frames: the number of sharp frames used as GT
        frames_used_to_synthesize: the number of frames used to be averaged
    """
    highfps_frames = os.listdir(video_path)
    highfps_frames.sort(key=lambda x: int(x.split(".")[0].split("-")[-1]))  # sort the read frames with time-order
    num_highfps_frames = len(highfps_frames) - (frames_used_to_synthesize // 2 + 1)  # to ensure each index can sample raw frames for blur synthesis

    half_total_frames = total_frames // 2

    # generate low-frame-rate (target blurry) video indices
    sharp_idx = np.arange(half_total_frames, num_highfps_frames, total_frames-1)

    # save the sharp-blurry frame pairs path
    sharp_path = os.path.join(output_video_path, "sharp", "raw")
    blur_path = os.path.join(output_video_path, "blur", "raw")
    os.makedirs(sharp_path, exist_ok=True)
    os.makedirs(blur_path, exist_ok=True)

    for i, idx in enumerate(sharp_idx):
        # GT frame sampling
        sharp_frame_list = []
        for j in range(idx - num_sharp_frames//2, idx - num_sharp_frames//2 + num_sharp_frames):
            frame = read_raw_img(os.path.join(video_path, highfps_frames[j]))
            sharp_frame_list.append(frame)
        sampled_frames = np.stack(sharp_frame_list, axis=0)
        sharp_frame = np.mean(sampled_frames, axis=0)

        # blurry frames sampling
        sampled_frame = []  # frames used to synthesize blurry frame
        if isinstance(frames_used_to_synthesize, list):
            num_frames_future_past = random.choice(frames_used_to_synthesize)
        else:
            num_frames_future_past = frames_used_to_synthesize
        for j in range(idx - num_frames_future_past//2, idx - num_frames_future_past//2 + num_frames_future_past):
            frame = read_raw_img(os.path.join(video_path, highfps_frames[j]))
            sampled_frame.append(frame)
        sampled_frame = np.stack(sampled_frame, axis=0)
        blur_frame = np.mean(sampled_frame, axis=0)

        # save blur-sharp raw pairs into tiff
        raw_sharp = Image.fromarray(sharp_frame.astype(np.uint16), mode="I;16")  # with shape (H, W)
        raw_blur = Image.fromarray(blur_frame.astype(np.uint16), "I;16")

        current_frame_name = f"{i:05d}.tiff"  # save generated raw dataset
        raw_sharp.save(os.path.join(sharp_path, current_frame_name))
        raw_blur.save(os.path.join(blur_path, current_frame_name))
        print(f"processing video {video_path} and {i}-th blur-sharp raw pairs done.")


def synthesize_rawblur_dataset(dataset_path, output_path):
    """
    Synthesize a raw dataset that contains multiple videos
    Args:
        dataset_path: the path of the raw dataset that contains multiple videos
        output_path: the path of the output rawblur dataset
    """
    videos = os.listdir(dataset_path)
    videos.sort()
    print(f"Total {len(videos)} are loaded to synthesize dataset ...")
    for v in videos:
        synthesize_rawblur_single_video(
            os.path.join(dataset_path, v),
            os.path.join(output_path, v),
            total_frames=33,
            num_sharp_frames=1,
            # frames_used_to_synthesize=[7, 9, 11, 13]
            frames_used_to_synthesize=11
        )
        print(f"Process raw video {v} done.")


def raw2rgb_ISP(raw_img_path, rgb_img_path=None, black_level=239):
    """
    transfer raw image into RGB image with ISP pipeline
    we use the ISP pipeline of the open-source library RawPy
    Args:
        raw_img_path: the path of the raw image
        rgb_img_path: the path of the output RGB image
        black_level: the black level of the raw image,
                     note that the BLC can be performed in reading raw data.
    """
    raw_buf = rawpy.imread(raw_img_path)
    rgb = raw_buf.postprocess(
        use_auto_wb=True,
        no_auto_bright=False,
        output_bps=16,
        user_black=black_level * 2**4,  # transfer the black-level into 16bit
    )
    rgb = (rgb.astype(np.float32) / 65535 * 255).astype(np.uint8)
    if rgb_img_path is None:
        return rgb[..., ::-1]
    else:
        cv2.imwrite(rgb_img_path, rgb[..., ::-1])
        print(f"transfer {raw_img_path} to {rgb_img_path}")


def rawblur_dataset_to_rgb(dataset_path, output_path, sigma=None, beta=None):
    """
    Transfer synthesized blur-sharp raw pairs into RGB format.
    We can add the G-P noise to the raw blurry image.
    Args:
        dataset_path: the path of the synthesized rawblur dataset
        output_path: the path of the output RGB dataset
        sigma: the sigma of the Gaussian noise
        beta: the beta of the Poisson noise
    """
    videos = os.listdir(dataset_path)
    videos.sort(key=lambda x: x.lower())
    print(f"Total {len(videos)} are loaded to synthesize dataset ...")
    for v in videos:
        sharp_video_path = os.path.join(dataset_path, v, "sharp")
        blur_video_path = os.path.join(dataset_path, v, "blur")

        sharp_rgb_imgs_path = os.path.join(output_path, v, "sharp")
        blur_rgb_imgs_path = os.path.join(output_path, v, "blur")
        os.makedirs(sharp_rgb_imgs_path, exist_ok=True)
        os.makedirs(blur_rgb_imgs_path, exist_ok=True)

        raw_imgs_name = os.listdir(sharp_video_path)
        raw_imgs_name.sort()
        video_length = len(raw_imgs_name)
        temp_tiff_image_path = os.path.join(sharp_rgb_imgs_path, "temp.tiff")
        for img in raw_imgs_name:
            img_name = img.split(".")[0]
            img_idx = int(img_name)

            img_indices = np.clip(np.arange(img_idx - 1, img_idx + 2), 1, video_length)  # temporal coherence

            raw_sharp = cv2.imread(os.path.join(sharp_video_path, img), -1)
            raw_blur = cv2.imread(os.path.join(blur_video_path, img), -1)
            H, W = raw_sharp.shape
            if beta:
                raw_blur = raw_blur.astype(np.float32)
                raw_blur = raw_blur + np.random.randn(H, W) * raw_blur * beta
            if sigma:
                raw_blur = raw_blur.astype(np.float32) + np.random.normal(0, sigma, [H, W])
                raw_blur = np.clip(raw_blur, 0, 2**16-1)
                raw_blur = raw_blur.astype(np.uint16)

            raw_sharp_blur = np.concatenate([raw_sharp, raw_blur], axis=0)

            temp_tiff = Image.fromarray(raw_sharp_blur, mode="I;16")
            temp_tiff.save(temp_tiff_image_path)

            rgb_sharp_blur = raw2rgb_ISP(
                temp_tiff_image_path,
                black_level=0,
            )
            rgb_sharp, rgb_blur = np.split(rgb_sharp_blur, 2, axis=0)

            # save sharp-blur paires
            cv2.imwrite(os.path.join(sharp_rgb_imgs_path, f"{img_name}.png"), rgb_sharp)
            cv2.imwrite(os.path.join(blur_rgb_imgs_path, f"{img_name}.png"), rgb_blur)
            print(f"Transfer raw imgs {os.path.join(sharp_video_path, img)} into RGB imgs {sharp_rgb_imgs_path}...")
        os.remove(temp_tiff_image_path)


if __name__ == "__main__":
    # 1 synthesize rawblurs
    # replace your own hyper-parameters
    synthesize_rawblur_dataset(
        RAW_DATASET_PATH,
        OUTPUT_PATH
    )

    # 2 transfer rawblurs into RGB images
    # replace your own hyper-parameters
    rawblur_dataset_to_rgb(
        RAWBLUR_DATASET_PATH,
        RGB_DATASET_PATH,
        sigma,
        beta
    )