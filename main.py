from fastapi import FastAPI, HTTPException
import requests
import os
import shutil
import uuid
from ppstructure.kie.predict_kie_token_ser import predict
import paddleclas
from PIL import Image
from pydantic import BaseModel
from paddleocr import PaddleOCR
import re

app = FastAPI()

clas_model = paddleclas.PaddleClas(model_name="text_image_orientation")


class Param(BaseModel):
    image_url: str


def pre_process(param):
    # Ensure the image directory exists
    image_dir = "./image/"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(image_dir, unique_filename)

    # Download image
    try:
        response = requests.get(param.image_url, stream=True)
        response.raise_for_status()
        # Save the image
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(response.raw, buffer)
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download image: {str(e)}"
        )
    finally:
        response.close()

    clas_result = next(clas_model.predict(input_data=image_path))
    angel = clas_result[0].get("label_names")[0]
    if angel == "90":
        rotate_and_replace(image_path, 90)
    elif angel == "180":
        rotate_and_replace(image_path, 180)
    elif angel == "270":
        rotate_and_replace(image_path, 270)

    return image_path


@app.post("/ai/predict_train_ticket")
async def predict_train_ticket(param: Param):
    return predict(0, pre_process(param))


@app.post("/ai/predict_id_card")
async def predict_id_card(param: Param):
    return predict(1, pre_process(param))


def ocr_id_card(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    result = ocr.ocr(image_path, cls=True)
    merged_string = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            merged_string += "".join(char for char in line[1][0] if char.isalnum())
    pattern = r"姓名(?P<姓名>\S+?)性别(?P<性别>\S+?)民族(?P<民族>\S+?)出生(?P<出生>\S+?)住址(?P<住址>.+?)公民身份号码(?P<公民身份号码>\S+)"
    match = re.search(pattern, merged_string)
    if match:
        result = {}
        result.setdefault("name", match.group("姓名"))
        result.setdefault("gender", match.group("性别"))
        result.setdefault("ethnicity", match.group("民族"))
        result.setdefault("birth", match.group("出生"))
        result.setdefault("address", match.group("住址"))
        result.setdefault("id_number", match.group("公民身份号码"))
        return result
    else:
        return None


def rotate_and_replace(image_path, angle):
    # 打开图片
    img = Image.open(image_path)

    # 旋转图片
    rotated_img = img.rotate(angle, expand=True)

    # 保存旋转后的图片并替换原图
    rotated_img.save(image_path)

    # 关闭图片
    img.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50010)
