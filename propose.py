import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import scipy.io as io

from lib.map import HeatMap
from lib.plot_annotation import PlotAnnotation


class RegionProposer:
    def __init__(self, path_to_images, path_to_annotations, path_to_enhanced_annotations, img_file_extension='jpg'):
        self.heatmap_obj = HeatMap()
        self.img_path = path_to_images
        self.annotation_path = path_to_annotations
        self.dest_annotation_path = path_to_enhanced_annotations
        self.img_file_extension = img_file_extension

    def _assert_path(self, path, error_message):
        assert os.path.exists(path), error_message

    def _display_image(self, image):
        # plt.axis('off')
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        plt.imshow(image)
        plt.show()

    def _display_images(self, images):
        plt.figure()
        # plt.figure(figsize=(20, 10))
        columns = 5
        for i, image in enumerate(images):
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            plt.imshow(image)
        plt.show()

    def save_to_mat(self, filename, detections):
        boxes = {'boxes': detections}
        proposals = {'proposals': boxes}
        io.savemat(filename, proposals)

    def save_image(self, image, path):
        cv2.imwrite(path, image)

    def propose(self):
        # Read each annotation
        for file_count, annotation_file in enumerate(os.listdir(self.annotation_path)):
            if os.path.isfile(os.path.join(self.annotation_path, annotation_file)):

                # Read the corresponding image
                file_name, _ = annotation_file.split('.')
                image_path = os.path.join(self.img_path, file_name + '.' + self.img_file_extension)
                self._assert_path(image_path, 'The corresponding image file for annotation not found at: ' + image_path)

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Parse the xml annotation
                annotation_xml = open(os.path.join(self.annotation_path, annotation_file), 'r')
                tree = ET.parse(annotation_xml)
                root = tree.getroot()
                intitial_annotation_count = len(root)

                # For each bb-annotation in annotation:
                boxes = []
                padding = 0
                for annotation in root.findall('./object'):
                    xmin = int(annotation.find('./bndbox/xmin').text)
                    ymin = int(annotation.find('./bndbox/ymin').text)
                    xmax = int(annotation.find('./bndbox/xmax').text)
                    ymax = int(annotation.find('./bndbox/ymax').text)

                    # Crop the patch
                    patch = image[ymin:ymax, xmin:xmax]

                    # Get the objectness
                    heat_map = self.heatmap_obj.get_map(patch)
                    heat_map = heat_map.data * ~heat_map.mask

                    # Remove the border in the detections
                    border = 2
                    temp = np.zeros_like(heat_map)
                    temp[border:-border, border:-border] = heat_map[border:-border, border:-border]
                    heat_map = temp

                    # Binary Map
                    heat_map[heat_map > 0] = 1
                    map_h, map_w = heat_map.shape

                    # Flood filling it
                    im_floodfill = heat_map.copy()
                    h, w = im_floodfill.shape[:2]
                    mask = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
                    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                    heat_map = heat_map | im_floodfill_inv

                    # Rejecting again if the number of disconnected components are > 3
                    im2, contours, hierarchy = cv2.findContours(heat_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    bounding_boxes = [cv2.boundingRect(c) for c in contours]
                    contour_area = [cv2.contourArea(c) for c in contours]

                    for (i, corners) in enumerate(bounding_boxes):
                        if contour_area[i] > 100:
                            (x, y, w, h) = corners
                            xmin_tight = (xmin + x - padding) if (x - padding) > 0 else xmin
                            ymin_tight = (ymin + y - padding) if (y - padding) > 0 else ymin
                            xmax_tight = (xmin + x + w + padding) if (x + w + padding) < map_w else xmin + map_w
                            ymax_tight = (ymin + y + h + padding) if (y + h + padding) < map_h else ymin + map_h

                            box = [xmin_tight, ymin_tight, xmax_tight, ymax_tight]
                            boxes.append(box)

                # Save the boxes to matlab file.
                self.save_to_mat(os.path.join(self.dest_annotation_path, file_name + '.' + self.img_file_extension+ '.mat'), boxes)

                # Plot annotation
                # p = PlotAnnotation(self.img_path, self.dest_annotation_path, file_name)
                # p.plot_annotation(boxes)
                # p.save_annotated_image(os.path.join(self.dest_annotation_path, file_name + '.' + self.img_file_extension+ '_annotated.jpg'))

                print 'Done with: ', file_count

    def unsupervised_propose(self):
        # Read each annotation
        for file_count, image_file in enumerate(os.listdir(self.img_path)):
            # Read the image
            file_name, _ = image_file.split('.')
            image_path = os.path.join(self.img_path, file_name + '.' + self.img_file_extension)
            self._assert_path(image_path, 'The  image file cannot read from: ' + image_path)

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # For each bb-annotation in annotation:
            boxes = []
            padding = 20

            xmin = 0
            ymin = 0

            # Get the objectness
            heat_map = self.heatmap_obj.get_map(image)
            heat_map = heat_map.data * ~heat_map.mask

            # Binary Map
            heat_map[heat_map > 0] = 1
            map_h, map_w = heat_map.shape

            # Flood filling it
            # im_floodfill = heat_map.copy()
            # h, w = im_floodfill.shape[:2]
            # mask = np.zeros((h + 2, w + 2), np.uint8)
            # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
            # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            # heat_map = heat_map | im_floodfill_inv

            # Finding Contours and bounding boxes
            im2, contours, hierarchy = cv2.findContours(heat_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            contour_area = [cv2.contourArea(c) for c in contours]

            for (i, corners) in enumerate(bounding_boxes):
                if contour_area[i] > 0:
                    (x, y, w, h) = corners

                    # Scaling
                    for padding in range(-200, 200, 10):
                        xmin_tight = (xmin + x - padding) if (x - padding) > 0 else xmin
                        ymin_tight = (ymin + y - padding) if (y - padding) > 0 else ymin
                        xmax_tight = (xmin + x + w + padding) if (x + w + padding) < map_w else xmin + map_w
                        ymax_tight = (ymin + y + h + padding) if (y + h + padding) < map_h else ymin + map_h

                        box = [xmin_tight, ymin_tight, xmax_tight, ymax_tight]
                        boxes.append(box)

                    # Translate x
                    for x_delta in range(-200, 200, 10):
                        xmin_tight = (xmin + x - x_delta) if (x - x_delta) > 0 else xmin
                        ymin_tight = (ymin + y) if y > 0 else ymin
                        xmax_tight = (xmin + x + w + x_delta) if (x + w + x_delta) < map_w else xmin + map_w
                        ymax_tight = (ymin + y + h) if (y + h) < map_h else ymin + map_h

                        box = [xmin_tight, ymin_tight, xmax_tight, ymax_tight]
                        boxes.append(box)

                    # Translate y
                    for y_delta in range(-200, 200, 10):
                        xmin_tight = (xmin + x) if x > 0 else xmin
                        ymin_tight = (ymin + y - y_delta) if (y - y_delta) > 0 else ymin
                        xmax_tight = (xmin + x + w) if (x + w) < map_w else xmin + map_w
                        ymax_tight = (ymin + y + h + y_delta) if (y + h + y_delta) < map_h else ymin + map_h

                        box = [xmin_tight, ymin_tight, xmax_tight, ymax_tight]
                        boxes.append(box)

            # Save the boxes to matlab file.
            self.save_to_mat(os.path.join(self.dest_annotation_path, file_name + '.' + self.img_file_extension+ '.mat'), boxes)

            # Plot annotation
            # self._display_image(heat_map)
            # p = PlotAnnotation(self.img_path, self.dest_annotation_path, file_name)
            # p.plot_annotation(boxes)
            # p.display_annotated_image()
            # p.save_annotated_image(os.path.join(self.dest_annotation_path, file_name + '.' + self.img_file_extension+ '_annotated.jpg'))

            print 'Done with: ', file_count
            print 'Len of boxes:', np.array(boxes).shape

    def getHeatMap(self, img_path, img_name):

        # Read the image
        image_path = os.path.join(img_path, img_name)
        self._assert_path(image_path, 'The  image file cannot read from: ' + image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Get the objectness
        heat_map = self.heatmap_obj.get_map(image)
        heat_map = heat_map.data * ~heat_map.mask
        objectness_heatmap = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

        file_name, _ = img_name.split('.')
        self.save_image(objectness_heatmap, os.path.join(img_path, file_name + '_result.png'))

        pickle_out = open(os.path.join(img_path, 'heat_map.pickle'), "wb")
        pickle.dump(heat_map, pickle_out)
        pickle_out.close()


        # # Binary Map
        # heat_map[heat_map > 0] = 1
        # map_h, map_w = heat_map.shape

        # Flood filling it
        # im_floodfill = heat_map.copy()
        # h, w = im_floodfill.shape[:2]
        # mask = np.zeros((h + 2, w + 2), np.uint8)
        # cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # heat_map = heat_map | im_floodfill_inv



if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    img_db_path = os.path.join('./data/images')
    annotation_path = os.path.join('./data/annotations')
    dest_annotation_path = os.path.join('./data/result')

    e = RegionProposer(img_db_path, annotation_path, dest_annotation_path)
    # e.unsupervised_propose()
    e.getHeatMap('/home/joseph', 'cat.jpg')
