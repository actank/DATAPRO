drop table if exists eval_predict;
create table eval_predict (
id string,
predict string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/src/prepare_data/test_data/predict.data' overwrite into table eval_predict;



drop table if exists eval_result;
create table eval_result as 
select
sum(label) as label_count,
sum(predict) as predict_count
from(
select
eval_predict.id,
f.start_geo_id as start_geo_id,
f.end_geo_id as end_geo_id,
f.create_date as create_date,
f.create_hour as create_hour,
label,
if(predict>0.9, 1, 0) as predict
from eval_predict
left join feature_data_all as f
on eval_predict.id = f.id
) t
group by
start_geo_id,end_geo_id,create_date,create_hour;


select
sum(error),
count(*),
sum(error) / count(*) as mae
from(
select
abs(label_count - predict_count) as error
from eval_result
) t;


