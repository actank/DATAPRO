#train val split
#暂时按照create_date字段，7月train 8月val来分


#xgboost模型的train/val去掉id类型特征，日期类型特征暂时不要
drop table if exists tree_train;
create table tree_train as
select
id,
--driver_id,
--member_id,
--create_date,
label,
create_hour,
estimate_money,
estimate_distance,
estimate_term,
--start_geo_id,
--end_geo_id,
start_loc1,
start_loc2,
start_loc3,
start_loc4,
start_loc5,
start_loc6,
start_loc7,
start_loc8,
start_loc9,
start_loc10,
end_loc1,
end_loc2,
end_loc3,
end_loc4,
end_loc5,
end_loc6,
end_loc7,
end_loc8,
end_loc9,
end_loc10,
--w_date,
w_hour,
code,
temperature,
feels_like,
pressure,
humidity,
visibility,
wind_direction,
wind_direction_degree,
wind_speed,
wind_scale
from feature_data_30
where create_date>='2017-07-01' and create_date<'2017-08-01'
;


drop table if exists tree_val;
create table tree_val as
select
id,
--driver_id,
--member_id,
--create_date,
label,
create_hour,
estimate_money,
estimate_distance,
estimate_term,
--start_geo_id,
--end_geo_id,
start_loc1,
start_loc2,
start_loc3,
start_loc4,
start_loc5,
start_loc6,
start_loc7,
start_loc8,
start_loc9,
start_loc10,
end_loc1,
end_loc2,
end_loc3,
end_loc4,
end_loc5,
end_loc6,
end_loc7,
end_loc8,
end_loc9,
end_loc10,
--w_date,
w_hour,
code,
temperature,
feels_like,
pressure,
humidity,
visibility,
wind_direction,
wind_direction_degree,
wind_speed,
wind_scale
from feature_data_30
where create_date>='2017-08-01' 
;

insert overwrite local directory '/home/hadoop/yjsdir/DATAPRO/UAICUP/src/prepare_data/train_data/' row format delimited fields terminated by ',' select * from tree_train;
insert overwrite local directory '/home/hadoop/yjsdir/DATAPRO/UAICUP/src/prepare_data/val_data/' row format delimited fields terminated by ',' select * from tree_val;

