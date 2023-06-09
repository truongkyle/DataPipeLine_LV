from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import Data_miningv2 as Data_Mining
import json



class Parameter(BaseModel): 
    period: Optional[List[str]] = None
    weekday: Optional[List[int]] = None
    district: Optional[List[str]] = None
    is_morning: Optional[List[int]] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
total_df = Data_Mining.processing_data()


@app.get("/data_mining/get_infor")
async def get_period_district():
    global total_df
    result = {
        'period_list': sorted(total_df.period.unique()),
        'district': sorted(total_df.district.unique()),
    }
    return result

@app.post("/data_mining/find_optimal_k")
async def find_optimal_k(parameter: Parameter):
    global total_df
    parameter = parameter.dict()
    period_df = Data_Mining.get_period_df(parameter, total_df)
    scaled_df = Data_Mining.MinMaxScale_function(period_df)
    K_number = Data_Mining.find_optimal_k(scaled_df)
    
    return {"K_number":K_number}

@app.post("/data_mining/dendograms")
async def get_dendograms(parameter: Parameter):
    pass


@app.post("/data_mining/K_means")
async def K_means(parameter: Parameter):
    global total_df
    seg_dicts = Data_Mining.get_seg_dicts_infor()
    total_df = Data_Mining.processing_data()
    parameter = parameter.dict()
    result_df, kmeans_labels = Data_Mining.get_results(parameter, total_df)

    group_result = result_df.groupby(["velocity_label"]).agg({"segment_id":["unique","count"], "tomtom_velocity":["max","min"], "duration_velocity":["max","min"]})
    result_list = []
    for item_label in result_df['velocity_label'].unique():
        seg_info_list = []
        for seg_id in group_result.segment_id.unique[item_label]:
            temp_df = result_df.loc[result_df.segment_id == seg_id].iloc[0]
            temp_dict = {
                "segment_id": int(seg_id),
                "position": [temp_df["lat"], temp_df["lng"]],
                "district": temp_df["district"]
            }
            seg_info_list.append(temp_dict)
        temp_item={
            "label": int(item_label),
            # "seg_id_list": group_result.segment_id.unique[item_label].astype(int).tolist(),
            "seg_info_list": seg_info_list,
            "velocity":{
                "max": int(group_result.tomtom_velocity["max"][item_label]),
                "min": int(group_result.tomtom_velocity["min"][item_label])
            },
            "duration":{
                "max": int(group_result.duration_velocity["max"][item_label]),
                "min": int(group_result.duration_velocity["min"][item_label])
            }    
        }
        result_list.append(temp_item)
    
    result = result_list
    return result

