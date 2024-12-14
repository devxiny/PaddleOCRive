# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import re
import PIL
from PIL import Image, ImageDraw, ImageFont

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import time

import tools.infer.utility as utility
from ppstructure.utility import parse_args
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

from paddleocr import PaddleOCR

logger = get_logger()


class SerPredictor(object):
    def __init__(self, args):
        self.ocr_engine = PaddleOCR(
            use_angle_cls=args.use_angle_cls,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            show_log=False,
            use_gpu=args.use_gpu,
            det_db_unclip_ratio=args.det_db_unclip_ratio,
        )

        pre_process_list = [
            {
                "VQATokenLabelEncode": {
                    "algorithm": args.kie_algorithm,
                    "class_path": args.ser_dict_path,
                    "contains_re": False,
                    "ocr_engine": self.ocr_engine,
                    "order_method": args.ocr_order_method,
                }
            },
            {"VQATokenPad": {"max_seq_len": 512, "return_attention_mask": True}},
            {"VQASerTokenChunk": {"max_seq_len": 512, "return_attention_mask": True}},
            {"Resize": {"size": [224, 224]}},
            {
                "NormalizeImage": {
                    "std": [58.395, 57.12, 57.375],
                    "mean": [123.675, 116.28, 103.53],
                    "scale": "1",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {
                "KeepKeys": {
                    "keep_keys": [
                        "input_ids",
                        "bbox",
                        "attention_mask",
                        "token_type_ids",
                        "image",
                        "labels",
                        "segment_offset_id",
                        "ocr_info",
                        "entities",
                    ]
                }
            },
        ]
        postprocess_params = {
            "name": "VQASerTokenLayoutLMPostProcess",
            "class_path": args.ser_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list, {"infer_mode": True})
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "ser", logger)

    def __call__(self, img):
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        if data[0] is None:
            return None, 0
        starttime = time.time()

        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        for idx in range(len(self.input_tensor)):
            self.input_tensor[idx].copy_from_cpu(data[idx])

        self.predictor.run()

        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        preds = outputs[0]

        post_result = self.postprocess_op(
            preds, segment_offset_ids=data[6], ocr_infos=data[7]
        )
        elapse = time.time() - starttime
        return post_result, data, elapse


train_ticket_args = parse_args()
train_ticket_args.det_db_unclip_ratio = 3.0
train_ticket_args.use_angle_cls = True
train_ticket_args.kie_algorithm = "LayoutXLM"
train_ticket_args.ser_model_dir = "./train_ticket_resource/ser_model"
train_ticket_args.det_model_dir = "./train_ticket_resource/det_model"
train_ticket_args.ser_dict_path = (
    "./train_ticket_resource/labels/class_list_id_card.txt"
)
train_ticket_args.vis_font_path = "./train_ticket_resource/fonts/simfang.ttf"
train_ticket_args.ocr_order_method = "tb-yx"
train_ticket_args.lang = "ch"
train_ticket_args.use_gpu = True
train_ticket_ser_predictor = SerPredictor(train_ticket_args)

id_card_args = parse_args()
id_card_args.det_db_unclip_ratio = 2.0
id_card_args.use_angle_cls = True
id_card_args.kie_algorithm = "LayoutXLM"
id_card_args.ser_model_dir = "./id_card_resource/ser_model"
id_card_args.det_model_dir = "./id_card_resource/det_model"
id_card_args.ser_dict_path = "./id_card_resource/labels/class_list_id_card.txt"
id_card_args.vis_font_path = "./id_card_resource/fonts/simfang.ttf"
id_card_args.ocr_order_method = "tb-yx"
id_card_args.lang = "ch"
id_card_args.use_gpu = True
id_card_ser_predictor = SerPredictor(id_card_args)


def predict(type, image_path):
    image_file_list = get_image_file_list(image_path)
    image_file = image_file_list[0]
    img, flag, _ = check_and_read(image_file)
    if not flag:
        img = cv2.imread(image_file)
        img = img[:, :, ::-1]
    if type == 0:
        ser_res, _, _ = train_ticket_ser_predictor(img)
        draw_ser_results(image_file, ser_res[0])
        return cast_train_ticket(ser_res[0])
    elif type == 1:
        ser_res, _, _ = id_card_ser_predictor(img)
        draw_ser_results(image_file, ser_res[0])
        return cast_id_card(ser_res[0])


def cast_train_ticket(list):
    result = {}
    for item in list:
        if item["pred_id"] != 0:
            pred_value = item["pred"]
            if "_" in pred_value:
                words = pred_value.split("_")
                camel_case_value = words[0].lower() + "".join(
                    word.title() for word in words[1:]
                )
            else:
                camel_case_value = pred_value
            if camel_case_value == "idInfo":
                id_number, name = split_id_name(item["transcription"])
                result.setdefault("id_num", id_number)
                result.setdefault("name", name)
            elif camel_case_value == "serialNumberInfo":
                result.setdefault(
                    "serial_number", remove_non_alphanumeric(item["transcription"])[:21]
                )
            elif camel_case_value == "dateTime":
                date, time = split_date_time(item["transcription"])
                result.setdefault("date", date)
                result.setdefault("time", time)
            else:
                result.setdefault(camel_case_value, item["transcription"])
    return result


def cast_id_card(list):
    result = {}
    address = ""
    for item in list:
        if item["pred_id"] != 0:
            pred_value = item["pred"].lower()
            if "_" in pred_value:
                words = pred_value.split("_")
                camel_case_value = words[0].lower() + "".join(
                    word.title() for word in words[1:]
                )
            else:
                camel_case_value = pred_value
            if camel_case_value.startswith("address"):
                address += item["transcription"]
            elif camel_case_value == "none":
                None
            else:
                result.setdefault(camel_case_value, item["transcription"])
    result.setdefault("address", address)
    return result


def split_date_time(input_string):
    match = re.search(
        r"(\d{4}年\d{2}月\d{2}日)(\d{2}:\d{2})", full_to_half(input_string)
    )

    if match:
        year_month_day = match.group(1)
        hour_minute = match.group(2)
        return year_month_day, hour_minute
    else:
        return None, None


def split_id_name(input_string):
    if len(input_string) > 18:
        return input_string[:18], input_string[18:]
    else:
        return None, None


def full_to_half(text):
    """将全角符号转换为半角符号"""
    half_text = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:  # 全角字符范围
            half_text.append(chr(code - 0xFEE0))  # 转换为半角字符
        elif code == 0x3000:  # 全角空格
            half_text.append(chr(0x0020))  # 转换为半角空格
        else:
            half_text.append(char)
    return "".join(half_text)


def remove_non_alphanumeric(s):
    # 使用正则表达式移除非字母和数字的字符
    return re.sub(r"[^a-zA-Z0-9]", "", s)


def draw_ser_results(
    image_file, ocr_results, font_path="./doc/fonts/simfang.ttf", font_size=14
):
    np.random.seed(2021)
    color = (
        np.random.permutation(range(255)),
        np.random.permutation(range(255)),
        np.random.permutation(range(255)),
    )
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx]) for idx in range(1, 255)
    }
    if isinstance(image_file, np.ndarray):
        image = Image.fromarray(image_file)
    elif isinstance(image_file, str) and os.path.isfile(image_file):
        image = Image.open(image_file).convert("RGB")
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]
        text = "{}: {}".format(ocr_info["pred"], ocr_info["transcription"])

        if "bbox" in ocr_info:
            # draw with ocr engine
            bbox = ocr_info["bbox"]
        else:
            # draw with ocr groundtruth
            bbox = trans_poly_to_bbox(ocr_info["points"])
        draw_box_txt(bbox, text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.7)

    img_res = np.array(img_new)
    cv2.imwrite(image_file, img_res)


def trans_poly_to_bbox(poly):
    x1 = np.min([p[0] for p in poly])
    x2 = np.max([p[0] for p in poly])
    y1 = np.min([p[1] for p in poly])
    y2 = np.max([p[1] for p in poly])
    return [x1, y1, x2, y2]


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    if int(PIL.__version__.split(".")[0]) < 10:
        tw = font.getsize(text)[0]
        th = font.getsize(text)[1]
    else:
        left, top, right, bottom = font.getbbox(text)
        tw, th = right - left, bottom - top

    start_y = max(0, bbox[0][1] - th)
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + th)],
        fill=(0, 0, 255),
    )
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)


if __name__ == "__main__":
    split_id_name("2024231998****156X赵所")
