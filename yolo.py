#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import csv
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS
import json
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/logoD.h5'
        self.anchors_path = 'model_data/logoD_anchors.txt'
        self.classes_path = 'model_data/logoD_classes.txt'
        self.output_path = 'output/'
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (608, 608) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def savebbox_txt(self, boxes, classes, scores, savebbox_path):
        with open(savebbox_path, 'w') as f:
            f.write('%d\n' % len(boxes))
            for i, c in reversed(list(enumerate(classes))):
                predicted_class = self.class_names[c]
                box = ' '.join(map(str, boxes[i].astype('int32')))
                score = scores[i]
                f.write(' '.join((box, predicted_class, '{:.4f}'.format(score))) + '\n')
        print('Image %s saved' % (savebbox_path))

    def savebbox_js(self, boxes, classes, scores, savebbox_path):
        with open(savebbox_path, 'w') as f:
            a = {}
            for i, c in reversed(list(enumerate(classes))):
                if self.class_names[c] not in a:
                    a[self.class_names[c]] = []
                b = {}
                b['score'] = str('{:.2f}'.format(scores[i]))
                b['top'],  b['left'],  b['bottom'],  b['right'] = boxes[i].astype('int32').astype('str')
                a[self.class_names[c]].append(b)
            json.dump(a, f)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)

        # from keras.models import Model
        # model1 = Model(inputs=self.yolo_model.input, outputs=self.yolo_model.get_layer('leaky_re_lu_58').output)
        print(self.yolo_model.summary())

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, output_json_prefix = None, show_bounding_box = True):
        start = time.time()

        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))


        if output_json_prefix is not None:
            head, tail = os.path.split(output_json_prefix)
            # savebbox_path = str('output/%s.txt' % (tail.split('.')[0]))
            savebbox_path = str(self.output_path + '%s.json' % (tail.split('.')[0]))
            self.savebbox_js(out_boxes, out_classes, out_scores, savebbox_path)

        if show_bounding_box:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = self.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # image drawing
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        end = time.time()
        print('Take %d seconds to detect.' % (end - start))
        return image

    def detect_video_frame(self, image, frame_id):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        frame_rows = []
        for i, c in reversed(list(enumerate(out_classes))):
            single_row = []
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # drawing
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            brand = predicted_class.split('-')[0]
            location = predicted_class.split('-')[1]
            w = right - left
            h = bottom - top
            area =  w*h
            pecentage = area*100/(image.width*image.height)
            central_x = (left + right)/2
            central_y = (top + bottom)/2
            if image.width//4 <central_x<= image.width*3//4 and image.height//4<central_y<= image.height*3//4:
                position = 'A'
            elif central_x<= image.width//2 and central_y<=image.height//2:
                position = 'B'
            elif central_x>image.width//2 and central_y<=image.height//2:
                position = 'C'
            elif central_x<=image.width//2 and central_y>image.height//2:
                position = 'D'
            elif central_x>image.width//2 and central_y>image.height//2:
                position = 'E'
            single_row.append(frame_id)
            single_row.append(brand)
            single_row.append(location)
            single_row.append(area)
            single_row.append(pecentage)
            single_row.append(central_x)
            single_row.append(central_y)
            single_row.append(position)
            single_row.append(score)
            frame_rows.append(single_row)
        end = timer()
        print(end - start)
        return image, frame_rows

    def detect_video(self,
                     video_path,
                     ms_per_frame = 1000,
                     save_video_with_boundingbox = True,
                     save_bounding_box_info=True,
                     show_result_live=True):

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open video")
        _, tail = os.path.split(video_path)


        if save_video_with_boundingbox:
            video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
            video_fps = vid.get(cv2.CAP_PROP_FPS)
            video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # isOutput = True if output_path != "" else False
            # if isOutput:
            output_path = tail.split('.')[0]+ "_with_BB." + tail.split('.')[1]
            out = cv2.VideoWriter(output_path, video_FourCC, 1/(ms_per_frame/1000), video_size)
            print('input video fps:{}'.format(video_fps))
            print('output video fps:{}'.format(1/(ms_per_frame/1000)))

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        if save_bounding_box_info:
            fid = open('%s.csv' % (tail.split('.')[0]), 'w', newline='')
            csvfile = csv.writer(fid, delimiter=',')
            csvfile.writerow([col_name.strip() for col_name in
                              'frame_id, brand, location, size, pecentage, x, y, position, confidence'.split(',')])

        # with open('%s.csv' % (tail.split('.')[0]), 'w', newline='') as fid:
        #     csvfile = csv.writer(fid, delimiter=',')
        #     csvfile.writerow([col_name.strip() for col_name in 'frame_id, brand, location, size, pecentage, x, y, position, confidence'.split(',')])
        count = 0
        while True:
            vid.set(cv2.CAP_PROP_POS_MSEC, ((count * ms_per_frame)))
            return_value, frame = vid.read()
            if return_value:
                try:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except:
                    image = Image.fromarray(frame)
            else:
                break
            image, frame_rows = self.detect_video_frame(image, count)

            result = np.asarray(image)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            if show_result_live:
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video_with_boundingbox:
                out.write(result)

            if save_bounding_box_info:
                csvfile.writerows(frame_rows)
            count = count + 1
            print('count:{}'.format(count))
        vid.release()
        out.release()
        self.close_session()

    def close_session(self):
        self.sess.close()

# def detect_video(yolo, video_path, show_result_live = True):
#     import cv2
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         raise IOError("Couldn't open video")
#     accum_time = 0
#     curr_fps = 0
#     fps = "FPS: ??"
#     prev_time = timer()
#     _, tail = os.path.split(video_path)
#     count = 0
#     while True:
#         return_value, frame = vid.read()
#         # print(cv2.CAP_PROP_POS_MSEC)
#         cv2.imshow("OpenCV", frame)
#         image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#         image = yolo.detect_image(image,
#                                   output_json_prefix = tail + 'frame'+ str(count),
#                                   show_bounding_box = show_result_live
#                                   )
#
#         if show_result_live:
#             result = np.asarray(image)
#             result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#             curr_time = timer()
#             exec_time = curr_time - prev_time
#             prev_time = curr_time
#             accum_time = accum_time + exec_time
#             curr_fps = curr_fps + 1
#             if accum_time > 1:
#                 accum_time = accum_time - 1
#                 fps = "FPS: " + str(curr_fps)
#                 curr_fps = 0
#             cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                         fontScale=0.50, color=(255, 0, 0), thickness=2)
#             cv2.namedWindow("result", cv2.WINDOW_NORMAL)
#             cv2.imshow("result", result)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         count += 1
#     yolo.close_session()

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image, img)
            r_image.show()
    yolo.close_session()

def detect_multiple_imgs(yolo, folder_path):
    for file in os.listdir(folder_path):
        image = Image.open(img)
        yolo.detect_image(image, file)
    yolo.close_session()




if __name__ == '__main__':
    # detect_multiple_imgs(YOLO())
    # detect_video(YOLO(), './test1.mp4')
    # detect_img(YOLO())

    y = YOLO()
    y.detect_video('./test1.mp4')