"""
Training part of my solution to The 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018
Goal of the competition was to create an algorithm to
automate nucleus detection from biomedical images.

author: Inom Mirzaev
github: https://github.com/mirzaevinom
"""

from config import *
import model as modellib
from pycocotools.coco import COCO
from PIL import Image
from pycocotools import mask as maskUtils

import h5py


class consid_dataset(utils.Dataset):
    
    
    def load_data(self, dataset_dir, subset, add_subset = None, class_ids=None):
        
        
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """
        
        
            

        coco = COCO("{}/{}/Annotations/{}.json".format(dataset_dir, subset, subset))
        image_dir = "{}/{}/Images".format(dataset_dir, subset)


        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                img_name= coco.imgs[i]['file_name'],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        new_i = len(image_ids) +1
        if add_subset == 'Mosaics':
            image_dir = "{}/{}/Images".format(dataset_dir, add_subset)
            # Add classes
            for id_, item in enumerate(os.listdir(image_dir)):
                i = new_i + int(id_)
                file_name_wo_ext = os.path.splitext(item)[0]
                filepath = os.path.join(image_dir, item)
                self.add_image('coco', image_id=i, 
                               path=filepath,
                               img_name=file_name_wo_ext)

        

        
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        get_name = image_info['img_name']
        
        if image_id > 1393:
        #if 'mosaic' not in get_name:
            get_name = image_info['img_name']
            path = os.path.join(os.path.abspath("../"),'considition-challenge', 'Mosaics/Masks', get_name + '.png')
            im_ = Image.open(path)
            im = im_.convert('RGB')
            channel = 3
            width, height = im.size #(width, height)
            mask_1_water = Image.new(mode = '1', size = (width, height))
            mask_2_bld = Image.new(mode = '1', size = (width, height))
            mask_3_road = Image.new(mode = '1', size = (width, height))
            
            mask_1_water_ = mask_1_water
            mask_2_bld_ = mask_2_bld
            mask_3_road_ = mask_3_road
            
            mask = []
            class_ids_ = []
            class_ids = []
            seen_class = []
            for x in range(width):
                for y in range(height):
                    r,g,b = im.getpixel((x,y))
                    # id=1 name= water    color= red    pixels= (255, 0, 0)
                    if (r == 255) and (g == 0) and (b == 0):
                        mask_1_water.putpixel((x,y),True)
                        class_ids_ = 1
                        if class_ids_ not in class_ids:
                            class_ids.append(class_ids_)
                    #id= 2 name= building color= yellow pixels= (255, 255, 0)
                    if (r == 255) and (g == 255) and (b == 0):
                        mask_2_bld.putpixel((x,y),True)
                        class_ids_ = 2
                        if class_ids_ not in class_ids:
                            class_ids.append(class_ids_)
                    #id= 3 name= road     color= pink   pixels= (255, 0, 255)
                    if (r == 255) and (g == 0) and (b == 255) :
                        mask_3_road.putpixel((x,y),True)
                        class_ids_ = 3
                        if class_ids_ not in class_ids:
                            class_ids.append(class_ids_)
            
            class_ids = list(set(list(class_ids)))
            mask = []
            for  class_ in class_ids:
                if class_ == 1:
                    mask.append(np.squeeze(mask_1_water))
                elif class_ == 2:
                    mask.append(np.squeeze(mask_2_bld))
                elif class_ == 3:
                    mask.append(np.squeeze(mask_3_road))
            
            mask = np.stack(mask, axis=2).astype(np.bool)
            #np.asarray(mask).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids #mask.astype(np.uint8), class_ids.astype(np.int8)

        if image_id <= 1393:
        #if 'mosaic' not in get_name:
        #if image_info["source"] == "coco":
            instance_masks = []
            class_ids = []
            annotations = self.image_info[image_id]["annotations"]
            # Build mask of shape [height, width, instance_count] and list
            # of class IDs that correspond to each channel of the mask.
            for annotation in annotations:
                class_id = self.map_source_class_id(
                    "coco.{}".format(annotation['category_id']))
                if class_id:
                    m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    # Is it a crowd? If so, use a negative class ID.
                    if annotation['iscrowd']:
                        # Use negative class ID for crowds
                        class_id *= -1
                        # For crowd masks, annToMask() sometimes returns a mask
                        # smaller than the given dimensions. If so, resize it.
                        if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                            m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                    instance_masks.append(m)
                    class_ids.append(class_id)

            # Pack instance masks into an array
            if class_ids:
                mask = np.stack(instance_masks, axis=2).astype(np.bool)
                class_ids = np.array(class_ids, dtype=np.int32)
                return mask, class_ids
            else:
                # Call super class to return an empty mask
                return super(consid_dataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        '''
        Return the path to the image.
        '''
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        import skimage
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def display_image(self, image_id): #theres a display_images at visuzalise
        '''
        
        '''
        # Load image
        import skimage
        img = skimage.io.imread(self.image_info[image_id]['path'])
        skimage.io.imshow(img)
        skimage.io.show()
    
    
    def image_data(self, image_id):
        '''
        
        '''
        info = self.image_info[image_id]
        return info
    
    
    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m