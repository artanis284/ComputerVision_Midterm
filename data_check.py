from pathlib import Path

from torchvision import datasets, transforms


def main():
    root = Path("data")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    cls_train = datasets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="category",
        transform=tf,
        download=True,
    )
    cls_test = datasets.OxfordIIITPet(
        root=str(root),
        split="test",
        target_types="category",
        transform=tf,
        download=True,
    )
    seg_train = datasets.OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    image, label = cls_train[0]
    _, mask = seg_train[0]
    print(f"classification trainval: {len(cls_train)}")
    print(f"classification test:     {len(cls_test)}")
    print(f"first image tensor:      {tuple(image.shape)}")
    print(f"first class id:          {label}")
    print(f"first mask size/mode:    {mask.size} / {mask.mode}")


if __name__ == "__main__":
    main()
