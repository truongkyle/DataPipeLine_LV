import Data_miningv2
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

def velocity_test1(parammeters):
    total_df = Data_miningv2.processing_data()
    label_encoder = LabelEncoder()
    total_df['base_LOS']= label_encoder.fit_transform(total_df['base_LOS'])
    velocity_k_mean_result, velocity_k_mean_label = Data_miningv2.get_results(parammeters, total_df)
    