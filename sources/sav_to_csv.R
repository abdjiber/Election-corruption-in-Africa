library('rio')
data = import("./data/merged_r5_data.sav")
export(data,'./data/merged_r5_data.csv')