from pathlib import Path
from configs.config import MainConfig
from confz import FileSource
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import load_detector, load_classificator, open_mapping, extract_crops
import pandas as pd
from itertools import repeat
import yaml

import pathlib
import PIL.Image as pim
from PIL.ExifTags import TAGS

import streamlit as st
import requests
from fastapi import FastAPI, File, UploadFile
from threading import Thread
import uvicorn
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class Payload(BaseModel):
    data_pth: str = ""

def base_ml():
    # Load main config
    main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load imgs from source dir

    pathes_to_imgs = [i for i in Path(main_config.src_dir).glob("*")
                      if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]
    # print(pathes_to_imgs)

    # Load mapping for classification task
    mapping = open_mapping(path_mapping=main_config.mapping)

    # Separate main config
    detector_config = main_config.detector
    classificator_config = main_config.classificator

    # Load models
    detector = load_detector(detector_config).to(device)
    classificator = load_classificator(classificator_config).to(device)
    # print(pathes_to_imgs)
    # Inference
    if len(pathes_to_imgs):

        list_predictions = []

        num_packages_det = np.ceil(len(pathes_to_imgs) / detector_config.batch_size).astype(np.int32)

        with torch.no_grad():
            for i in tqdm(range(num_packages_det), colour="green"):
                # Inference detector
                batch_images_det = pathes_to_imgs[detector_config.batch_size * i:
                                                  detector_config.batch_size * (1 + i)]
                results_det = detector(batch_images_det,
                                       iou=detector_config.iou,
                                       conf=detector_config.conf,
                                       imgsz=detector_config.imgsz,
                                       verbose=False,
                                       device=device)

                if len(results_det) > 0:
                    # Extract crop by bboxes
                    dict_crops = extract_crops(results_det, config=classificator_config)

                    # Inference classificator
                    for img_name, batch_images_cls in dict_crops.items():
                        # if len(batch_images_cls) > classificator_config.batch_size:
                        num_packages_cls = np.ceil(len(batch_images_cls) / classificator_config.batch_size).astype(
                            np.int32)
                        for j in range(num_packages_cls):
                            batch_images_cls = batch_images_cls[classificator_config.batch_size * j:
                                                                classificator_config.batch_size * (1 + j)]
                            logits = classificator(batch_images_cls.to(device))
                            probabilities = torch.nn.functional.softmax(logits, dim=1)
                            top_p, top_class_idx = probabilities.topk(1, dim=1)

                            # Locate torch Tensors to cpu and convert to numpy
                            top_p = top_p.cpu().numpy().ravel()
                            top_class_idx = top_class_idx.cpu().numpy().ravel()

                            class_names = [mapping[top_class_idx[idx]] for idx, _ in enumerate(batch_images_cls)]

                            list_predictions.extend([[name, cls, prob] for name, cls, prob in
                                                     zip(repeat(img_name, len(class_names)), class_names, top_p)])

        # Create Dataframe with predictions
        table = pd.DataFrame(list_predictions, columns=["image_name", "class_name", "confidence"])
        # table.to_csv("table.csv", index=False) # Раскомментируйте, если хотите увидеть результаты предсказания
        # нейронной сети по каждому найденному объекту

        agg_functions = {
            'class_name': ['count'],
            "confidence": ["mean"]
        }
        groupped = table.groupby(['image_name', "class_name"]).agg(agg_functions)
        img_names = groupped.index.get_level_values("image_name").unique()

        final_res = []

        for img_name in img_names:
            groupped_per_img = groupped.query(f"image_name == '{img_name}'")
            max_num_objects = groupped_per_img["class_name", "count"].max()
            # max_confidence = groupped_per_img["class_name", "confidence"].max()
            statistic_by_max_objects = groupped_per_img[groupped_per_img["class_name", "count"] == max_num_objects]

            if len(statistic_by_max_objects) > 1:
                # statistic_by_max_mean_conf = statistic_by_max_objects.reset_index().max().values
                statistic_by_max_mean_conf = statistic_by_max_objects.loc[
                    [statistic_by_max_objects["confidence", "mean"].idxmax()]]
                final_res.extend(statistic_by_max_mean_conf.reset_index().values)
            else:
                final_res.extend(statistic_by_max_objects.reset_index().values)

        # groupped.to_csv("table_agg.csv", index=True) # Раскомментируйте, если хотите увидеть результаты аггрегации

        final_table = pd.DataFrame(final_res, columns=["image_name", "class_name", "count", "confidence"])

        final_table = pd.DataFrame(
            {'raw_pth': [pathes_to_imgs[i].parent for i in range(len(pathes_to_imgs))]}
        ).join(final_table)

        final_table.to_csv("table_final.csv", index=False)


def print_exif_data(image_path):
    image = pim.open(image_path)
    exif = image.getexif()
    exif_data = exif.get_ifd(0x8769)
    data = {}
    for tag_id in exif_data:
        tag = TAGS.get(tag_id, tag_id)
        content = exif_data.get(tag_id)
        data[tag] = content
    if 'DateTimeOriginal' in data.keys():
        return data['DateTimeOriginal']


def final_folder_creation(bl_fit_tab):
    time_holder = []
    class_holder = []
    bl_fit_tab = bl_fit_tab.dropna()
    # name_folder,class,date_registration_start,date_registration_end,count
    print(os.listdir())
    if 'fin_res.csv' in os.listdir():
        print('exists')
        print()
        fin_res = pd.read_csv('fin_res.csv')
    else:
        print('doesnt exists')
        print()
        fin_res = pd.DataFrame(
            columns=['name_folder', 'class', 'date_registration_start', 'date_registration_end', 'count'])

    first_time = 0
    ani_count = 0

    for row in bl_fit_tab.itertuples():
        time_flag = False
        class_flag = False
        print(row)

        path_to_img = pathlib.Path(row.raw_pth)

        time_info = (print_exif_data(path_to_img.joinpath(row.image_name)))
        time_info = pd.to_datetime(time_info, exact=0, format='%y:%m:%d %H:%M:%S')

        # registration logic
        if len(time_holder) > 0:
            # check conditions
            time_flag = True if time_info - time_holder[-1] >= pd.Timedelta(minutes=30) else False
            class_flag = True if row.class_name == class_holder[-1] else False
            # if time delta < 30 and class is the same, registration declined

            if not time_flag and class_flag:
                print('same class in 30 min')
                first_time = time_info if first_time == 0 else first_time
                ani_count = max(ani_count, row.count)
                print(ani_count)

                fin_res.drop(fin_res.tail(1).index, inplace=True)  # drop last n rows

                fin_res = fin_res._append({
                    'name_folder': path_to_img.parts[-1],
                    'class': row.class_name,
                    'date_registration_start': first_time,
                    'date_registration_end': time_info,
                    'count': ani_count,
                },
                    ignore_index=1)

            # in other case registration is done
            else:
                first_time = time_info
                ani_count = row.count
                print('first_time', first_time, 'ani_count', ani_count)

                # add em to vault
                time_holder.append(time_info)
                class_holder.append(row.class_name)

                fin_res = fin_res._append({
                    'name_folder': path_to_img.parts[-1],
                    'class': row.class_name,
                    'date_registration_start': time_info,
                    'date_registration_end': time_info,
                    'count': row.count,
                },
                    ignore_index=1)
            # print()
        else:
            first_time = time_info
            ani_count = row.count
            print('first_time_ever', first_time, 'ani_count', ani_count)
            time_holder.append(time_info)
            class_holder.append(row.class_name)
            fin_res = fin_res._append({
                'name_folder': path_to_img.parts[-1],
                'class': row.class_name,
                'date_registration_start': time_info,
                'date_registration_end': time_info,
                'count': row.count,
            },
                ignore_index=1)

    fin_res.to_csv("fin_res.csv", index=False)
    return time_holder, class_holder


def sub_table():
    final_folder_creation(pd.read_csv('table_final.csv'))


def starter():
    main_config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
    pathes_to_imgs = [i for i in Path(main_config.src_dir).glob("*")
                      if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]
    # print(pathes_to_imgs)

    folds_in_path = [i for i in Path(main_config.src_dir).glob("*") if i not in pathes_to_imgs]
    print(folds_in_path)
    files_in_folds_in_path = [len(os.listdir(i)) for i in folds_in_path]

    # check if folder w folders or folder and run
    if len(folds_in_path) and sum(files_in_folds_in_path) != 0:
        for folder_inside in folds_in_path:
            with open(os.path.join("configs", "config.yml"), mode='r') as conf_f:
                con_f = yaml.safe_load(conf_f)

            # print(con_f)
            con_f['src_dir'] = f'{folder_inside}'
            print(con_f)

            with open(os.path.join("configs", "config.yml"), "w") as f:
                yaml.dump(con_f, f, sort_keys=False)

            base_ml()
            sub_table()

    else:
        base_ml()
        sub_table()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Создание FastAPI приложения
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=['*']
)


@app.post("/run_model/")
async def run_model(file: Payload):
    print(file)
    with open(os.path.join("configs", "config.yml"), mode='r') as conf_f:
        con_f = yaml.safe_load(conf_f)

    # print(con_f)
    con_f['src_dir'] = f'{file.data_pth}'
    # print(con_f)

    with open(os.path.join("configs", "config.yml"), "w") as f:
        yaml.dump(con_f, f, sort_keys=False)

    starter()

    # img = Image.open(file.file)
    # results = pd.read_csv('fin_res.csv')
    result_path = f"fin_res.csv"
    # results[0].save(filename=result_image_path)
    return FileResponse(result_path, media_type='text/csv')


# Функция для запуска API сервера
def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Запуск API сервера в отдельном потоке
api_thread = Thread(target=run_api, daemon=True)
api_thread.start()

# Создание Streamlit приложения
st.title('Registrations')


def get_result(file):
    # files = {'path': file,{'Content-Type': 'text/plain'}}
    print(file)
    response = requests.post('http://127.0.0.1:8000/run_model/',  json={"data_pth":file})
    print(response)
    if response.status_code == 500:
        st.error('Ошибка в работе модели')
        return
    st.dataframe(pd.read_csv('fin_res.csv'))


users_path = st.text_input("Введите к папке с файлами и нажмите Enter")
# if users_path:
#    with open('users_path.txt', "w") as f:
#        f.write(users_path)
file = users_path

if file and os.path.isdir(file):
    correct_dir = True
    st.write('есть такая папка')
    st.button(label='Запустить модель', on_click=get_result, args=(file,))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



if __name__ == '__st_app_1__':
    uvicorn.run(host='0.0.0.0', port=8000, app="st_app_1:app")
