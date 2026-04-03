from vesskel.hrf import HRFDataset, preprocess_segmentation
from vesskel.thin import lee94_thin


def main():
    dataset = HRFDataset("HRF")

    image, seg, mask, info = dataset.load_sample(0)
    print(f"Sample: {info['name']}")
    print(f"Image shape: {image.shape}")
    print(f"Segmentation shape: {seg.shape}")
    print(f"Vessel pixels: {seg.sum()}")

    cleaned = preprocess_segmentation(seg, mask)
    print(f"After preprocessing: {cleaned.sum()} pixels")

    skeleton = lee94_thin(cleaned)
    print(f"Skeleton pixels: {skeleton.sum()}")


if __name__ == "__main__":
    main()
