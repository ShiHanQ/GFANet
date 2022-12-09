
from utils.transform import *
from torch.utils.data.dataset import Dataset


class DealDataset(Dataset):
    def __init__(self, args, images_path, labels_path, transform, num_classes, mode, normVal=1.10):

        self.classweight_name = args.dataset
        self.NUM_CLASSES = num_classes
        self.mode = mode
        self.normVal = normVal

        self.images_path_list = images_path
        self.labels_path_list = labels_path

        self.trans = transform

        self.valid_classes = [0, 255]

        self.class_names = ['background', 'object']

        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

    def __getitem__(self, index):

        image_path = self.images_path_list[index]
        label_path = self.labels_path_list[index]

        image = Image.open(image_path).convert("RGB")

        label = Image.open(label_path).convert("L")

        if image.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))

        label = np.array(label, dtype=np.uint8)
        label = self.encode_segmap(label)
        label = Image.fromarray(label)

        image, label = self.trans(image=image, target=label)

        return image, label

    def __len__(self):
        return len(self.images_path_list)

    def encode_segmap(self, mask):
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


def dataloader(args, images_path, labels_path, num_classes, mode):
    data_transform = {
        "train": Compose([randomcrop(size=(224, 320)),
                          Randomflip_Rotate(p=0.5, degrees=30),
                          ToTensor(),
                          Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),

        "val": Compose([resize(size=(224, 320)), ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    }
    data_set = DealDataset(args=args, images_path=images_path, transform=data_transform[mode],
                           labels_path=labels_path, num_classes=num_classes, mode=mode)

    return data_set

