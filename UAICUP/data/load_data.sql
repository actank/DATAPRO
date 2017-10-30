create table train_July(
id string,
driver_id string,
member_id string,
create_date string,
create_hour string,
status int,
estimate_money float,
estimate_distance float,
estimate_term float,
start_geo_id string,
end_geo_id string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/data/train_July.csv' overwrite into table train_July;
create table train_Aug(
id string,
driver_id string,
member_id string,
create_date string,
create_hour string,
status int,
estimate_money float,
estimate_distance float,
estimate_term float,
start_geo_id string,
end_geo_id string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/data/train_Aug.csv' overwrite into table train_Aug;


--这里用人工补充整点数据weather_fill.csv代替weather.csv
create table weather(
w_date string,
w_text string,
code int,
temperature int,
feels_like int,
pressure int,
humidity int,
visibility float,
wind_direction string,
wind_direction_degree int,
wind_speed float,
wind_scale int)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/data/weather_fill.csv' overwrite into table weather;

create table poi(
id string,
loc1 int,
loc2 int,
loc3 int,
loc4 int,
loc5 int,
loc6 int,
loc7 int,
loc8 int,
loc9 int,
loc10 int)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/data/poi_trans.csv' overwrite into table poi;

create table test_id_Aug_agg_public5k(
test_id int,
start_geo_id string,
end_geo_id string,
create_date string,
create_hour int)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
load data local inpath '/home/hadoop/yjsdir/DATAPRO/UAICUP/data/test_id_Aug_agg_public5k.csv' overwrite into table test_id_Aug_agg_public5k;
