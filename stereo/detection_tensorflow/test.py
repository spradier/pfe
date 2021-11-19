from os import walk

filenames = next(walk("train/image"), (None, None, []))[2]
print(len(filenames))


 def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        onlyfiles = [f for f in listdir(self.img_dir) if isfile(join(self.img_dir, f))]
        return len(onlyfiles)

    def __getitem__(self, filename):
        image_name = filename + ".png"
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path)
        path = 'dataset_xml_format/' + filename + ".xml"
        tree = ET.parse(path)
        root = tree.getroot()
        labels = [root[6][4][0].text, root[6][4][1].text, root[6][4][2].text, root[6][4][3].text]
        #convert labels in tensor 
        boxes = torch.as_tensor(labels, dtype=torch.float32)
        #there is only one class so I have to give the class 1 to my dataset 
        num_objs = len(labels)

        labels_type = torch.ones((num_objs,), dtype=torch.int64) # ne peut pas marcher avec num_objs 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels_type
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target   

