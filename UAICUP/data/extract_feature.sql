##四个维度，member，driver，location，weather
##数据清洗放在各个子表中进行，保证干净的feature_data_30数据
##当天的weather特征，
##统计7月的driver/member的统计类型特征,
##统计7月的member-start_geo_id-end_geo_id组合特征
##统计7月的driver-start_geo_id-end_geo_id组合特征
##统计7月driver每天的成交/拒绝/成交率特征
##统计7月member每天的成交/拒绝/成交率特征
##统计7月member-create_hour特征
##统计7月周末-start_geo_id-end_geo_id组合特征
##统计7月周末-driver-start_geo_id-end_geo_id组合特征
##统计7月member周末平均成交次数
##统计7月member工作日平均每天成交单数
##统计weather-是否周末组合特征
##周末要单独处理，单独特征，或者单独模型
##时间窗口搞一个7天预测7天的模型进行滑动扩大训练集验证集，和30天预测7天的模型来融合

#同时训练lr/ffm(带id类型)，xgboost不带id类型，两个模型组合，两个模型的类别特征都要编码
#这样长短期模型每个模型又有lr／ffm／xgboost组合，共四个模型stacking，
#最后上wide&deep


#30天模型，train选为train_July，val选为train_Aug
#xgboost模型最后要去掉id类型
drop table if exists feature_data_30;
create table feature_data_30 as 
select
union_data.id,
driver_id,
member_id,
create_date,
create_hour,
if(status=2, 1, 0) as label,
estimate_money,
estimate_distance,
estimate_term,
start_geo_id,
end_geo_id,
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
w_date,
w_hour,
code,
temperature,
feels_like,
pressure,
humidity,
visibility,
wind_direction_cate as wind_direction,
wind_direction_degree,
wind_speed,
wind_scale
from (
select *
from train_July
UNION
select *
from train_Aug
) as union_data
--有些instance没有起始或终止poi，在这里用innerjoin过滤这部分数据
inner join (
select id,
loc1 as start_loc1,
loc2 as start_loc2,
loc3 as start_loc3,
loc4 as start_loc4,
loc5 as start_loc5,
loc6 as start_loc6,
loc7 as start_loc7,
loc8 as start_loc8,
loc9 as start_loc9,
loc10 as start_loc10
from poi
) as feature_start_poi on union_data.start_geo_id = feature_start_poi.id
inner join (
select id,
loc1 as end_loc1,
loc2 as end_loc2,
loc3 as end_loc3,
loc4 as end_loc4,
loc5 as end_loc5,
loc6 as end_loc6,
loc7 as end_loc7,
loc8 as end_loc8,
loc9 as end_loc9,
loc10 as end_loc10
from poi
) as feature_end_poi on union_data.end_geo_id = feature_end_poi.id
left join (
select
from_unixtime(unix_timestamp(w_date, 'yyyy-M-d HH:mm'), 'yyyy-MM-dd') as w_date,
from_unixtime(unix_timestamp(w_date, 'yyyy-M-d HH:mm'), 'HH') as w_hour,
code,
temperature,
feels_like,
pressure,
humidity,
visibility,
wind_direction_cate,
wind_direction_degree,
wind_speed,
wind_scale
from weather
left join
(select 
--编号
row_number() over (order by wind_direction) as wind_direction_cate,
wind_direction 
from weather 
where w_date!='date' 
group by wind_direction) as weather_label_encoder
on weather.wind_direction = weather_label_encoder.wind_direction
) as feature_weather on union_data.create_date = feature_weather.w_date and union_data.create_hour=feature_weather.w_hour
;


