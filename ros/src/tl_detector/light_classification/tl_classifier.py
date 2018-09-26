from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import os
import tensorflow as tf
from utilities import label_map_util
from utilities import visualization_utils as vis_util
import cv2

class TLClassifier(object):
    def __init__(self, is_site):
        # set default value for no detection
        self.current_light = TrafficLight.UNKNOWN
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = curr_dir + '/../../../../image_classification_model/'
        if is_site:
            model = model_dir + '/protobuf/frozen_inference_graph_ssd_real.pb'
        else:
            model = model_dir + '/protobuf/frozen_inference_graph_ssd_sim.pb'

        num_classes = 4
        labels_file = model_dir + '/label_map.pbtxt'
        label_map = label_map_util.load_labelmap(labels_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Input tensor
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents the confidence level for each object
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.image_classified = None
        print("Frozen graph loaded! model: {}".format(model))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        min_score_thresh = 0.20
        merged_scores = {}
        for i in range(len(classes)):
            class_id = classes[i]
            score = scores[i]
            if score > min_score_thresh:
                if class_id in merged_scores:
                    merged_scores[class_id] += score
                else:
                    merged_scores[class_id] = score

        # Find the class id of the max score
        max_score = 0.0
        selected_class_id = -1
        min_merged_score_thresh = .50
        for class_id, score in merged_scores.items():
            if score > max_score and score > min_merged_score_thresh:
                max_score = score
                selected_class_id = class_id
      
        class_name = 'UNKNOWN'
        self.current_light = TrafficLight.UNKNOWN
        if selected_class_id != -1:
            class_name = self.category_index[selected_class_id]['name']
            if class_name == 'Red':
                self.current_light = TrafficLight.RED
            elif class_name == 'Green':
                self.current_light = TrafficLight.GREEN
            elif class_name == 'Yellow':
                self.current_light = TrafficLight.YELLOW

        print('TL_CLassifier:: class_name: {}, max score: {}'.format(class_name, max_score))

        vis_util.visualize_boxes_and_labels_on_image_array(
            image, boxes, classes, scores, self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        self.image_classified = image

        return self.current_light
