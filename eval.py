class RotatedDateset:
    def __init__(self, dataset, angle):
        self.dataset = dataset
        self.angle = angle
    def __getitem__(self, index):
        img, target = self.dataset[index]
        img = img.rotate(self.angle)
        return img, target
    def __len__(self):
        return len(self.dataset)