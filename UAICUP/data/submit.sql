#启发式策略召回，召回过去30天有动作的用户，构造7天submit数据进行预测，类似于构造train_Aug
#本质上是围绕订单扩展特征，预测什么样的订单在给定的时间地点是否会发生。
#(阿里移动推荐大赛这里是启发式召回过去3天有交互无购买的user-item对补充特征作为ins进行预测submit)
#推荐大赛的限制条件是user-item对，

#新的召回策略：召回过去每周一到周日的订单，分别预测周一到周日。。。这不就是过去一个月的么。。。
#剔除8-1/8-7已经有时间点的数据
#尝试回归

#按照每一个详细特征回归订单数，最后按照start_poi,end_poi,create_date,create_hour进行聚合
#按照过去一个月的每个和预测test表按时间相同进行联立的样本进行抽取，预测八月相同时间的进行聚合，提交，然后求平均
--取7月4日开始后4周的同时间数据作为召回ins，进行预测，最后结果订单量/4
--接下来尝试用7月25-7月31日数据来直接预测

drop table if exists submit_data;
set  hive.jobname.length=10;
create table submit_data as
select 
test_id,
--id,
--create_date,
--start_geo_id,
--end_geo_id,
create_week,
create_hour,
--缺失值用众数填充
if(avg_estimate_money is NULL,34.00,avg_estimate_money) as avg_estimate_money,
--缺失值用平均值填充
if(avg_estimate_distance is NULL,11039.6,avg_estimate_distance) as avg_estimate_distance,
--缺失值用平均值填充
if(avg_estimate_term is NULL,19.40,avg_estimate_term) as avg_estimate_term,
--poi起始点或终点没有的，用众数填充，只是没有poi周边信息的也暂时用950poi众数填充
if(ps.poi_cate is NULL,'950',ps.poi_cate) as start_geo_cate,
if(pe.poi_cate is NULL,'950',pe.poi_cate) as end_geo_cate,
if(ps.loc1 is NULL,'3',ps.loc1) as start_loc1,
if(ps.loc2 is NULL,'63',ps.loc2) as start_loc2,
if(ps.loc3 is NULL,'272',ps.loc3) as start_loc3,
if(ps.loc4 is NULL,'7',ps.loc4) as start_loc4,
if(ps.loc5 is NULL,'49',ps.loc5) as start_loc5,
if(ps.loc6 is NULL,'334',ps.loc6) as start_loc6,
if(ps.loc7 is NULL,'898',ps.loc7) as start_loc7,
if(ps.loc8 is NULL,'189',ps.loc8) as start_loc8,
if(ps.loc9 is NULL,'310',ps.loc9) as start_loc9,
if(ps.loc10 is NULL,'206',ps.loc10) as start_loc10,
if(pe.loc1 is NULL,'3',pe.loc1) as end_loc1,
if(pe.loc2 is NULL,'63',pe.loc2) as end_loc2,
if(pe.loc3 is NULL,'272',pe.loc3) as end_loc3,
if(pe.loc4 is NULL,'7',pe.loc4) as end_loc4,
if(pe.loc5 is NULL,'49',pe.loc5) as end_loc5,
if(pe.loc6 is NULL,'334',pe.loc6) as end_loc6,
if(pe.loc7 is NULL,'898',pe.loc7) as end_loc7,
if(pe.loc8 is NULL,'189',pe.loc8) as end_loc8,
if(pe.loc9 is NULL,'310',pe.loc9) as end_loc9,
if(pe.loc10 is NULL,'206',pe.loc10) as end_loc10,
--route_driver_num as route_driver_num,
--if(feature_start_poi_cluster.cluster_id is NULL, 0, feature_start_poi_cluster.cluster_id) as start_cluster_id,
--if(feature_end_poi_cluster.cluster_id is NULL, 0, feature_end_poi_cluster.cluster_id) as end_cluster_id,
--w_date,
if(w.w_hour is NULL,create_hour,w.w_hour) as w_hour,
--天气缺失用当天上午的天气填充
if(w.code is NULL,w1.code,w.code) as code,
if(w.temperature is NULL,w1.temperature,w.temperature) as temperature,
if(w.feels_like is NULL,w1.feels_like,w.feels_like) as feels_like,
if(w.pressure is NULL,w1.pressure,w.pressure) as pressure,
if(w.humidity is NULL,w1.humidity,w.humidity) as humidity,
if(w.visibility is NULL,w1.visibility,w.visibility) as visibility,
if(w.wind_direction_cate is NULL,w1.wind_direction_cate,w.wind_direction_cate) as wind_direction_cate,
if(w.wind_direction_degree is NULL,w1.wind_direction_degree,w.wind_direction_degree) as wind_direction_degree,
if(w.wind_speed is NULL,w1.wind_speed,w.wind_speed) as wind_speed,
if(w.wind_scale is NULL,w1.wind_scale,w.wind_scale) as wind_scale
from
(select
test_id,
start_geo_id,
end_geo_id,
create_date,
pmod(datediff(create_date,'1920-01-01')-3,7) as create_week,
create_hour
from test_id_aug_agg_public5k
)t
left join weather_feature as w on t.create_date = w.w_date and t.create_hour = w.w_hour
left join poi_feature as ps on t.start_geo_id = ps.id
left join poi_feature as pe on t.end_geo_id = pe.id
left join route_feature as r on t.start_geo_id = r.start_geo_id and t.end_geo_id = r.end_geo_id
left join weather_feature as w1 on t.create_date = w1.w_date and t.create_hour = (w1.w_hour - 12)
--left join poi_cluster as feature_start_poi_cluster on t.start_geo_id = feature_start_poi_cluster.id
--left join poi_cluster as feature_end_poi_cluster on t.end_geo_id = feature_end_poi_cluster.id
--left join route_feature3 as feature_route3 on t.start_geo_id = feature_route3.start_geo_id and t.end_geo_id = feature_route3.end_geo_id
where test_id is not NULL
order by test_id
;

insert overwrite local directory '/home/hadoop/yjsdir/DATAPRO/UAICUP/src/prepare_data/submit_data/' row format delimited fields terminated by ',' select * from submit_data;




