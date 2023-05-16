from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import Data_Mining
import json



class Parameter(BaseModel): 
    period: Optional[List[str]] = None
    weekday: Optional[List[int]] = None
    district: Optional[List[str]] = None
    is_morning: Optional[List[int]] = None


app = FastAPI()
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
    print(period_df.info())
    scaled_df = Data_Mining.MinMaxScale_function(period_df)
    K_number = Data_Mining.find_optimal_k(scaled_df)
    
    return {"K_number":K_number}

@app.post("/data_mining/dendograms")
async def get_dendograms(parameter: Parameter):
    pass

@app.post("/data_mining/K_means")
async def K_means(parameter: Parameter):
    global total_df
    total_df = Data_Mining.processing_data()
    parameter = parameter.dict()
    result_df = Data_Mining.get_results(parameter, total_df)
    group_result = result_df.groupby(["label"]).agg({"segment_id":["unique","count"], "tomtom_velocity":["max","min"], "duration":["max","min"]})
    result_list = []
    for item_label in result_df['label'].unique():
        temp_item={
            "label": int(item_label),
            "seg_id_list": group_result.segment_id.unique[item_label].astype(int).tolist(),
            "velocity":{
                "max": int(group_result.tomtom_velocity["max"][item_label]),
                "min": int(group_result.tomtom_velocity["min"][item_label])
            },
            "duration":{
                "max": int(group_result.duration["max"][item_label]),
                "min": int(group_result.duration["min"][item_label])
            }    
        }
        result_list.append(temp_item)
    
    result = {"data": result_list}
    return result

