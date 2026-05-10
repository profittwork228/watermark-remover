import argparse
import sys
import cv2

from watermark_remover import WatermarkDetector, WatermarkRemover, VideoProcessor
from watermark_remover.utils import get_video_info, visualize_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark-remover",
        description="Remove watermarks from video files using OpenCV inpainting.",
    )
    parser.add_argument("input", type=str, help="Input video path")
    parser.add_argument("output", type=str, help="Output video path")
    parser.add_argument(
        "--method",
        choices=["telea", "ns", "blur"],
        default="telea",
        help="Removal method (default: telea)",
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Manual watermark region: x y w h",
    )
    parser.add_argument(
        "--logo",
        type=str,
        default=None,
        metavar="LOGO_PATH",
        help="Path to logo template image for template matching",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=30,
        dest="sample_frames",
        help="Number of frames to sample for auto-detection (default: 30)",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=5,
        dest="mask_dilation",
        help="Kernel size for mask dilation (default: 5)",
    )
    parser.add_argument(
        "--preview-mask",
        action="store_true",
        dest="preview_mask",
        help="Save mask visualization as mask_preview.png before processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        info = get_video_info(args.input)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Input: {args.input}")
        print(f"  Resolution : {info['width']}x{info['height']}")
        print(f"  FPS        : {info['fps']:.2f}")
        print(f"  Frames     : {info['frame_count']}")
        print(f"  Duration   : {info['duration_seconds']:.2f}s")
        print(f"Output: {args.output}")
        print(f"Method: {args.method}")

    detector = WatermarkDetector()
    remover = WatermarkRemover()

    mask = None
    region = tuple(args.region) if args.region else None

    if args.preview_mask:
        cap = cv2.VideoCapture(args.input)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Cannot read first frame for mask preview.", file=sys.stderr)
            sys.exit(1)

        if region is not None:
            preview_mask = detector.detect_by_region(first_frame, region)
        elif args.logo:
            preview_mask = detector.detect_logo(first_frame, args.logo)
            if preview_mask is None:
                print("Warning: Logo not found in first frame; using static detection for preview.")
                preview_mask = detector.detect_static_watermark(args.input, sample_frames=args.sample_frames)
        else:
            preview_mask = detector.detect_static_watermark(args.input, sample_frames=args.sample_frames)

        from watermark_remover.utils import dilate_mask
        preview_mask = dilate_mask(preview_mask, kernel_size=args.mask_dilation)
        preview_overlay = visualize_mask(first_frame, preview_mask)
        cv2.imwrite("mask_preview.png", preview_overlay)
        print("Saved mask preview to mask_preview.png")

    def progress_callback(frame_num: int, total: int) -> None:
        pct = (frame_num / total * 100) if total > 0 else 0.0
        print(f"Processing frame {frame_num}/{total} ({pct:.1f}%)")

    processor = VideoProcessor()

    try:
        processor.process(
            input_path=args.input,
            output_path=args.output,
            mask=mask,
            detector=detector,
            remover=remover,
            method=args.method,
            region=region,
            logo_template=args.logo,
            progress_callback=progress_callback,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Done. Output written to: {args.output}")


if __name__ == "__main__":
    main()
