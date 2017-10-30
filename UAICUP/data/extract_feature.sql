##六个维度的特征，route，time，member，driver，poi，weather
##合并训练集验证集测试集，知识集和训练集相同，提交集就是测试集，提交集的知识集选为训练集＋验证集
##label为给定线路和时间情况下的订单需求量
##数据清洗放在各个子表中进行，保证干净的feature_data_all数据
##价格／时长／距离用平均价格／平均时长／平均距离代替
##知识集的时间段采用训练集的时间段不变，不然太复杂了，最后submit的时候可以扩大覆盖到测试集作为知识集
##价格分桶
##done:人工补齐weather数据
##done:poi聚类 无效特征
##done:统计每条路线的driver个数
##模型stacking
##两个poi周围是否有公交站地铁站
##每个路线每个天气下的订单数
##统计7月该week的driver/member的统计类型特征,
##统计7月的member-start_geo_id-end_geo_id组合特征
##统计7月的driver-start_geo_id-end_geo_id组合特征
##统计7月driver每天的成交/拒绝/成交率特征
##统计7月member每天的成交/拒绝/成交率特征
##统计7月member-create_hour特征
##统计7月周末-start_geo_id-end_geo_id组合特征
##统计7月周末-driver-start_geo_id-end_geo_id组合特征
##统计7月member周末平均成交次数
##统计7月member工作日平均每天成交单数
##统计上下午平均单数
##统计晚上加班人群组合特征（聚类）
##统计周末加班人群组合特征（聚类）
##统计中午吃饭／晚饭应酬人群（聚类）／平均单数划分
##统计weather-是否周末组合特征
##周末要单独处理，单独特征，或者单独模型
##时间窗口搞一个7天预测7天的模型进行滑动扩大训练集验证集，和30天预测7天的模型来融合

#同时训练lr/ffm(带id类型)，lightgbm，xgboost不带id类型，两个模型组合，两个模型的类别特征都要编码
#这样长短期模型每个模型又有lr／ffm／xgboost组合，共四个模型stacking，
#最后上wide&deep
#xgboost模型最后要去掉id类型



#回归
drop table if exists feature_data_all;
create table feature_data_all as 
select
union_data.status as status,
union_data.create_date as create_date,
union_data.create_hour as create_hour,
union_data.create_week as create_week,
union_data.start_geo_id as start_geo_id,
union_data.end_geo_id as end_geo_id,
feature_route.avg_estimate_money as avg_estimate_money,
feature_route.avg_estimate_distance as avg_estimate_distance,
feature_route.avg_estimate_term as avg_estimate_term,
--feature_route2.avg_route_week_order_num as avg_route_week_order_num,
--feature_week.avg_week_order_num as avg_week_order_num,
--feature_week1.avg_week_route_num as avg_week_route_num,
--feature_week2.avg_week_route_order_num as avg_week_route_order_num,
feature_start_poi.poi_cate as start_geo_cate,
feature_end_poi.poi_cate as end_geo_cate,
feature_start_poi.loc1 as start_loc1,
feature_start_poi.loc2 as start_loc2,
feature_start_poi.loc3 as start_loc3,
feature_start_poi.loc4 as start_loc4,
feature_start_poi.loc5 as start_loc5,
feature_start_poi.loc6 as start_loc6,
feature_start_poi.loc7 as start_loc7,
feature_start_poi.loc8 as start_loc8,
feature_start_poi.loc9 as start_loc9,
feature_start_poi.loc10 as start_loc10,
feature_end_poi.loc1 as end_loc1,
feature_end_poi.loc2 as end_loc2,
feature_end_poi.loc3 as end_loc3,
feature_end_poi.loc4 as end_loc4,
feature_end_poi.loc5 as end_loc5,
feature_end_poi.loc6 as end_loc6,
feature_end_poi.loc7 as end_loc7,
feature_end_poi.loc8 as end_loc8,
feature_end_poi.loc9 as end_loc9,
feature_end_poi.loc10 as end_loc10,
--feature_start_poi_cluster.cluster_id as start_cluster_id,
--feature_end_poi_cluster.cluster_id as end_cluster_id,
feature_route3.route_driver_num as route_driver_num,
w_date,
w_hour,
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
from union_data
inner join poi_feature as feature_start_poi on union_data.start_geo_id = feature_start_poi.id
inner join poi_feature as feature_end_poi on union_data.end_geo_id = feature_end_poi.id
left join route_feature as feature_route on union_data.start_geo_id = feature_route.start_geo_id and union_data.end_geo_id = feature_route.end_geo_id
--left join route_feature2 as feature_route2 on union_data.start_geo_id = feature_route2.start_geo_id and union_data.end_geo_id = feature_route2.end_geo_id and union_data.create_week = feature_route2.create_week
--left join week_feature as feature_week on union_data.create_week = feature_week.create_week
--left join week_feature1 as feature_week1 on union_data.create_week = feature_week1.create_week
left join week_feature2 as feature_week2 on union_data.create_week = feature_week2.create_week
left join weather_feature on union_data.create_date = weather_feature.w_date and union_data.create_hour=weather_feature.w_hour
--left join poi_cluster as feature_start_poi_cluster on union_data.start_geo_id = feature_start_poi_cluster.id
--left join poi_cluster as feature_end_poi_cluster on union_data.end_geo_id = feature_end_poi_cluster.id
left join route_feature3 as feature_route3 on union_data.start_geo_id = feature_route3.start_geo_id and union_data.end_geo_id = feature_route3.end_geo_id
;

#其他feature表单独写在后面
#拆分成weather_feature，poi_feature两个feature表，主表通过start_geo_id,end_geo_id,create_date,create_hour来联立

------------------------------union_data订单样本抽取表--------------------------------
drop table if exists union_data;
create table union_data as 
select
if(status=2 or status=0, 1, 0) as status,
create_date,
pmod(datediff(create_date,'1920-01-01')-3,7) as create_week,
create_hour,
start_geo_id,
end_geo_id,
estimate_money,
-----------money分桶
estimate_distance,
-----------distance分桶
estimate_term
-----------行程时间分桶
from train_July
UNION 
select
if(status=2 or status=0, 1, 0) as status,
create_date,
pmod(datediff(create_date,'1920-01-01')-3,7) as create_week,
create_hour,
start_geo_id,
end_geo_id,
estimate_money,
estimate_distance,
estimate_term
from train_Aug
;

------------------------------route特征表--------------------------------------------
--所有去重路线
drop table if exists route_all;
create table route_all as 
select
start_geo_id,end_geo_id
from(
select
start_geo_id,
end_geo_id
from union_data
UNION
select
start_geo_id,
end_geo_id 
from
test_id_aug_agg_public5k
) t
group by start_geo_id,end_geo_id;


drop table if exists route_feature;
create table route_feature as
select
start_geo_id as start_geo_id,
end_geo_id as end_geo_id,
round(avg_estimate_money,2) as avg_estimate_money,
round(avg_estimate_distance,2) as avg_estimate_distance,
round(avg_estimate_term,2) as avg_estimate_term
from(
select
start_geo_id,
end_geo_id,
AVG(estimate_money) OVER(partition by start_geo_id,end_geo_id) as avg_estimate_money,
AVG(estimate_distance) OVER(partition by start_geo_id,end_geo_id) as avg_estimate_distance,
AVG(estimate_term) OVER(partition by start_geo_id,end_geo_id) as avg_estimate_term
from union_data
)t
group by start_geo_id,end_geo_id,avg_estimate_money,avg_estimate_distance,avg_estimate_term
;

#7月该路线该week该hour平均订单需求量
#空值太多，废弃特征 
drop table if exists route_feature1;
create table route_feature1 as
select
start_geo_id as start_geo_id,
end_geo_id as end_geo_id,
create_week,
create_hour,
--7月4日至7月24日共三周，每个分组里每个小时都重复了3次，因此/3
round(sum(status)/3,2) as avg_week_hour_order_num
from union_data
--训练知识集取7月4日至7月24日，注意不要穿越
where create_date>='2017-07-04' and create_date<'2017-07-25'
group by start_geo_id,end_geo_id,create_week,create_hour
;

--这里用平均值补齐avg_route_week_order_num的数据，null数据占3%
--每条路线每星期一到星期日的平均订单数
--这个特征需要仔细计算
drop table if exists route_feature2;
create table route_feature2 as 
select 
tt.start_geo_id,
tt.end_geo_id,
tt.create_week,
--如果该路线没有平均每曜日每路线订单数，则用0填充，因为该路线很可能很少有人走，不该用平均值填充
if(avg_route_week_order_num is NULL,0,avg_route_week_order_num) as avg_route_week_order_num
from(
select 
union_data.start_geo_id,
union_data.end_geo_id,
union_data.create_week,
avg_route_week_order_num
--从union_data表取所有的路线数据
from union_data
left join(
select
start_geo_id as start_geo_id,
end_geo_id as end_geo_id,
create_week,
--7月4日至7月24日共三周，因此/3
round(sum(status)/3,2) as avg_route_week_order_num
from union_data
--训练知识集取7月4日至7月24日，注意不要穿越
where create_date>='2017-07-04' and create_date<'2017-07-25'
group by start_geo_id,end_geo_id,create_week
) t on union_data.start_geo_id = t.start_geo_id and union_data.end_geo_id = t.end_geo_id and union_data.create_week = t.create_week
group by 
union_data.start_geo_id,
union_data.end_geo_id,
union_data.create_week,
avg_route_week_order_num
) tt
;

--每条路线知识集中的去重driver数
drop table if exists route_feature3;
create table route_feature3 as
select 
route_all.start_geo_id as start_geo_id,
route_all.end_geo_id as end_geo_id,
if(count(distinct driver_id) is not NULL,count(distinct driver_id),0) as route_driver_num
from route_all
left join (
select
start_geo_id,
end_geo_id,
driver_id
from train_July
where train_July.create_date>='2017-07-04' and train_July.create_date<'2017-07-25'
) t
on route_all.start_geo_id = t.start_geo_id and route_all.end_geo_id = t.end_geo_id
group by route_all.start_geo_id,route_all.end_geo_id;




------------------------------time特征表---------------------------------------------
--每曜日平均订单数
drop table if exists week_feature;
create table week_feature as 
select 
create_week,
round(sum(status)/3,2) as avg_week_order_num
from union_data 
where create_date>='2017-07-04' and create_date<'2017-07-25' 
group by create_week
;

--每曜日平均路线数
drop table if exists week_feature1;
create table week_feature1 as
select 
create_week,
round(count(distinct start_geo_id,end_geo_id)/3,2) as avg_week_route_num
from union_data 
where create_date>='2017-07-04' and create_date<'2017-07-25' 
group by create_week
;
--每曜日所有路线的平均订单数
drop table if exists week_feature2;
create table week_feature2 as 
select
week_feature.create_week as create_week,
round(avg_week_order_num/avg_week_route_num,2) as avg_week_route_order_num
from week_feature
left join week_feature1
on week_feature.create_week=week_feature1.create_week
;
------------------------------weather特征表------------------------------------------
drop table if exists weather_feature;
create table weather_feature as
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
from weather as w
left join
(select 
--编号
row_number() over (order by wind_direction) as wind_direction_cate,
wind_direction 
from weather 
where w_date!='date' 
group by wind_direction
) as t
on w.wind_direction = t.wind_direction
where split(w_date,':')[1]='00'
;
-----------------------------poi特征表-------------------------------------------------
drop table if exists poi_feature;
create table poi_feature as 
select poi.id,
loc1,
loc2,
loc3,
loc4,
loc5,
loc6,
loc7,
loc8,
loc9,
loc10,
poi_cate
from poi
left join(
select
id,
row_number() over(order by id) as poi_cate
from poi
) t on poi.id=t.id;

create table poi_cluster(
id string,
cluster_id int
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/src/poi_cluster.csv' overwrite into table poi_cluster;


