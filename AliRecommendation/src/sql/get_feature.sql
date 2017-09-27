use tb;



---------------------------------------------训练集-----------------------------------------------------------
--train label
drop table if exists train_label_10;
create table train_label_10 as 
select
    user_id, item_id,
    case behavior_type when 4 then '1' else '0' end as label
from
    tianchi_fresh_comp_train_user l
where time>='2014-12-17 00' and time<'2014-12-18 00';

--点击，收藏，加车行为数统计
drop table if exists 1_item_features_10;
create table 1_item_features_10 as 
select
    l.item_id as item_id,
    if(item_category is not null, item_category, 0) as item_category,
    sum(case when behavior_type=1 then 1 else 0 end) as click_num_10,
    sum(case when behavior_type=2 then 1 else 0 end) as favorite_num_10,
    sum(case when behavior_type=3 then 1 else 0 end) as cart_num_10,
    sum(case when behavior_type=4 then 1 else 0 end) as order_num_10
from
    tianchi_fresh_comp_train_user l
where
    time>='2014-12-07 00' and time<'2014-12-17 00'
group by l.item_id, item_category;

--转化率，加车率，收藏率
drop table if exists 2_item_features_10;
create table 2_item_features_10 as 
select
    item_id, 
    sum(case when behavior_type=4 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as cvr_10,
    sum(case when behavior_type=3 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pcart_10,
    sum(case when behavior_type=2 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pfavorite_10
from 
    tianchi_fresh_comp_train_user
where
    time>='2014-12-07 00' and time<'2014-12-17 00'
group by item_id;


--用户-商品特征
--平均每天对商品行为数，对商品总行为数
drop table if exists 1_ui_features_10;
create table 1_ui_features_10 as
select
    user_id,item_id,
    sum(case when behavior_type=1 then 1 else 0 end) as ui_1,
    sum(case when behavior_type=2 then 1 else 0 end) as ui_2,
    sum(case when behavior_type=3 then 1 else 0 end) as ui_3,
    sum(case when behavior_type=4 then 1 else 0 end) as ui_4,
    sum(case when behavior_type=1 then 1 else 0 end)/10 as ui_5,
    sum(case when behavior_type=2 then 1 else 0 end)/10 as ui_6,
    sum(case when behavior_type=3 then 1 else 0 end)/10 as ui_7,
    sum(case when behavior_type=4 then 1 else 0 end)/10 as ui_8
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-07 00' and time<'2014-12-17 00'
group by user_id, item_id;

--用户对商品最后一(缺二，三，七天内，因为一次计算内存不够，放到多个表里)天内是否有行为
drop table if exists 2_ui_features_10;
create table 2_ui_features_10 as 
select
    user_id,item_id,
    sum(case when (behavior_type=1 and datediff(to_date('2014-12-17'), to_date(time))=1 ) then 1 else 0 end) as ui_9,
    sum(case when (behavior_type=2 and datediff(to_date('2014-12-17'), to_date(time))=1 ) then 1 else 0 end) as ui_10,
    sum(case when (behavior_type=3 and datediff(to_date('2014-12-17'), to_date(time))=1 ) then 1 else 0 end) as ui_11,
    sum(case when (behavior_type=4 and datediff(to_date('2014-12-17'), to_date(time))=1 ) then 1 else 0 end) as ui_12
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-07 00' and time<'2014-12-17 00'
group by user_id, item_id;

--用户过去七天动作数，用户过去七天平均每天行为数
drop table if exists 1_user_features_10;
create table 1_user_features_10 as 
select
    user_id,
    sum(case when (behavior_type=1 and datediff(to_date('2014-12-17'), to_date(time))<=7) then 1 else 0 end) as u_1,
    sum(case when (behavior_type=2 and datediff(to_date('2014-12-17'), to_date(time))<=7) then 1 else 0 end) as u_2,
    sum(case when (behavior_type=3 and datediff(to_date('2014-12-17'), to_date(time))<=7) then 1 else 0 end) as u_3,
    sum(case when (behavior_type=4 and datediff(to_date('2014-12-17'), to_date(time))<=7) then 1 else 0 end) as u_4,
    sum(case when (datediff(to_date('2014-12-17'), to_date(time))<=7) then 1 else 0 end)/7 as u_5
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-07 00' and time<'2014-12-17 00'
group by user_id;


--合并feature_window长度为10的item／ui／user特征，构造train样本，
drop table if exists train_10;
create table train_10 as
select
    l.user_id,l.item_id,f1.item_category,
    if(click_num_10>0, click_num_10, 0) as click_num_10,
    if(favorite_num_10>0, favorite_num_10, 0) as favorite_num_10,
    if(cart_num_10>0, cart_num_10, 0) as cart_num_10,
    if(order_num_10>0, order_num_10, 0) as order_num_10,
    if(cvr_10>0, cvr_10, 0) as cvr_10,
    if(pcart_10>0, pcart_10, 0) as pcart_10,
    if(pfavorite_10>0, pfavorite_10, 0) as pfavorite_10,
    if(ui_1>0, ui_1, 0) as ui_1,
    if(ui_2>0, ui_2, 0) as ui_2,
    if(ui_3>0, ui_3, 0) as ui_3,
    if(ui_4>0, ui_4, 0) as ui_4,
    if(ui_5>0, ui_5, 0) as ui_5,
    if(ui_6>0, ui_6, 0) as ui_6,
    if(ui_7>0, ui_7, 0) as ui_7,
    if(ui_8>0, ui_8, 0) as ui_8,
    if(ui_9>0, ui_9, 0) as ui_9,
    if(ui_10>0, ui_9, 0) as ui_10,
    if(ui_11>0, ui_11, 0) as ui_11,
    if(ui_12>0, ui_12, 0) as ui_12,
    if(u_1>0, u_1, 0) as u_1,
    if(u_2>0, u_2, 0) as u_2,
    if(u_3>0, u_3, 0) as u_3,
    if(u_4>0, u_4, 0) as u_4,
    if(u_5>0, u_5, 0) as u_5,
    label as label
from train_label_10 l
left join 1_item_features_10 f1 on l.item_id=f1.item_id
left join 2_item_features_10 f2 on l.item_id=f2.item_id
left join 1_ui_features_10 f3 on l.item_id=f3.item_id and l.user_id=f3.user_id
left join 2_ui_features_10 f4 on l.item_id=f4.item_id and l.user_id=f4.user_id
left join 1_user_features_10 f5 on l.user_id=f5.user_id;
--group by l.user_id,l.item_id,item_category,click_num_10,favorite_num_10,cart_num_10,order_num_10,cvr_10,pcart_10,pfavorite_10,label;


--合并多个滑动窗口的train(train_w1, train_w2...,train_w7)，尝试多个feature_window的train(train_1,....train_10)


--------------------------------------------训练集结束---------------------------------------------------------------

--------------------------------------------验证集-----------------------------------------------------------------
--val label
drop table if exists val_label_10;
create table val_label_10 as 
select
    item_id,
    case behavior_type when 4 then '1' else '0' end as label
from
    tianchi_fresh_comp_train_user l
where time>='2014-12-18 00' and time<'2014-12-19 00';

--点击，收藏，加车行为数统计
drop table if exists 1_item_features_10;
create table 1_item_features_10 as 
select
    l.item_id as item_id,
    sum(case when behavior_type=1 then 1 else 0 end) as click_num_10,
    sum(case when behavior_type=2 then 1 else 0 end) as favorite_num_10,
    sum(case when behavior_type=3 then 1 else 0 end) as cart_num_10,
    sum(case when behavior_type=4 then 1 else 0 end) as order_num_10
from
    tianchi_fresh_comp_train_user l
where
    time>='2014-12-08 00' and time<'2014-12-18 00'
group by l.item_id;

--转化率，加车率，收藏率
drop table if exists 2_item_features_10;
create table 2_item_features_10 as 
select
    item_id, 
    sum(case when behavior_type=4 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as cvr_10,
    sum(case when behavior_type=3 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pcart_10,
    sum(case when behavior_type=2 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pfavorite_10
from 
    tianchi_fresh_comp_train_user
where
    time>='2014-12-08 00' and time<'2014-12-18 00'
group by item_id;

--合并feature_window长度为10的特征，构造train样本
drop table if exists val_10;
create table val_10 as
select
    m.item_id as item_id,
    m.user_id as user_id,
    click_num_10 as click_num_10,
    favorite_num_10 as favorite_num_10,
    cart_num_10 as cart_num_10,
    order_num_10 as order_num_10,
    cvr_10 as cvr_10,
    pcart_10 as pcart_10,
    pfavorite_10 as pfavorite_10,
    label as label
from tianchi_fresh_comp_train_user m
left join 1_item_features_10 f1 on m.item_id=f1.item_id
left join 2_item_features_10 f2 on m.item_id=f2.item_id
left join val_label_10 l on m.item_id=l.item_id;
--------------------------------------------验证集结束--------------------------------------------------------------


--------------------------------------------提交集------------------------------------------------------------------
--采用19号之前两天有交互无购买的作为预测作用对，这个需要调整，相当于模型之前的规则过滤
drop table if exists test_label_10;
create table test_label_10 as 
select
    user_id, item_id
from
    tianchi_fresh_comp_train_user l
where time>='2014-12-17 00' and time<'2014-12-19 00' and behavior_type!=4;


--点击，收藏，加车行为数统计
drop table if exists test_1_item_features_10;
create table test_1_item_features_10 as 
select
    l.item_id as item_id,
    if(item_category is not null, item_category, 0) as item_category,
    sum(case when behavior_type=1 then 1 else 0 end) as click_num_10,
    sum(case when behavior_type=2 then 1 else 0 end) as favorite_num_10,
    sum(case when behavior_type=3 then 1 else 0 end) as cart_num_10,
    sum(case when behavior_type=4 then 1 else 0 end) as order_num_10
from
    tianchi_fresh_comp_train_user l
where
    time>='2014-12-09 00' and time<'2014-12-19 00'
group by l.item_id, item_category;

--转化率，加车率，收藏率
drop table if exists test_2_item_features_10;
create table test_2_item_features_10 as 
select
    item_id, 
    sum(case when behavior_type=4 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as cvr_10,
    sum(case when behavior_type=3 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pcart_10,
    sum(case when behavior_type=2 then 1 else 0 end)/sum(case when behavior_type=1 then 1 else 0 end) as pfavorite_10
from 
    tianchi_fresh_comp_train_user
where
    time>='2014-12-09 00' and time<'2014-12-19 00'
group by item_id;


--用户-商品特征
--平均每天对商品行为数，对商品总行为数
drop table if exists test_1_ui_features_10;
create table test_1_ui_features_10 as
select
    user_id,item_id,
    sum(case when behavior_type=1 then 1 else 0 end) as ui_1,
    sum(case when behavior_type=2 then 1 else 0 end) as ui_2,
    sum(case when behavior_type=3 then 1 else 0 end) as ui_3,
    sum(case when behavior_type=4 then 1 else 0 end) as ui_4,
    sum(case when behavior_type=1 then 1 else 0 end)/10 as ui_5,
    sum(case when behavior_type=2 then 1 else 0 end)/10 as ui_6,
    sum(case when behavior_type=3 then 1 else 0 end)/10 as ui_7,
    sum(case when behavior_type=4 then 1 else 0 end)/10 as ui_8
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-09 00' and time<'2014-12-19 00'
group by user_id, item_id;

--用户对商品最后一(缺二，三，七天内，因为一次计算内存不够，放到多个表里)天内是否有行为
drop table if exists test_2_ui_features_10;
create table test_2_ui_features_10 as 
select
    user_id,item_id,
    sum(case when (behavior_type=1 and datediff(to_date('2014-12-19'), to_date(time))=1 ) then 1 else 0 end) as ui_9,
    sum(case when (behavior_type=2 and datediff(to_date('2014-12-19'), to_date(time))=1 ) then 1 else 0 end) as ui_10,
    sum(case when (behavior_type=3 and datediff(to_date('2014-12-19'), to_date(time))=1 ) then 1 else 0 end) as ui_11,
    sum(case when (behavior_type=4 and datediff(to_date('2014-12-19'), to_date(time))=1 ) then 1 else 0 end) as ui_12
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-09 00' and time<'2014-12-19 00'
group by user_id, item_id;

--用户过去七天动作数，用户过去七天平均每天行为数
drop table if exists test_1_user_features_10;
create table test_1_user_features_10 as 
select
    user_id,
    sum(case when (behavior_type=1 and datediff(to_date('2014-12-19'), to_date(time))<=7) then 1 else 0 end) as u_1,
    sum(case when (behavior_type=2 and datediff(to_date('2014-12-19'), to_date(time))<=7) then 1 else 0 end) as u_2,
    sum(case when (behavior_type=3 and datediff(to_date('2014-12-19'), to_date(time))<=7) then 1 else 0 end) as u_3,
    sum(case when (behavior_type=4 and datediff(to_date('2014-12-19'), to_date(time))<=7) then 1 else 0 end) as u_4,
    sum(case when (datediff(to_date('2014-12-19'), to_date(time))<=7) then 1 else 0 end)/7 as u_5
from tianchi_fresh_comp_train_user 
where 
    time>='2014-12-09 00' and time<'2014-12-19 00'
group by user_id;


--合并feature_window长度为10的item／ui／user特征，构造train样本，
drop table if exists test_10;
create table test_10 as
select
    l.user_id,l.item_id,f1.item_category,
    if(click_num_10>0, click_num_10, 0) as click_num_10,
    if(favorite_num_10>0, favorite_num_10, 0) as favorite_num_10,
    if(cart_num_10>0, cart_num_10, 0) as cart_num_10,
    if(order_num_10>0, order_num_10, 0) as order_num_10,
    if(cvr_10>0, cvr_10, 0) as cvr_10,
    if(pcart_10>0, pcart_10, 0) as pcart_10,
    if(pfavorite_10>0, pfavorite_10, 0) as pfavorite_10,
    if(ui_1>0, ui_1, 0) as ui_1,
    if(ui_2>0, ui_2, 0) as ui_2,
    if(ui_3>0, ui_3, 0) as ui_3,
    if(ui_4>0, ui_4, 0) as ui_4,
    if(ui_5>0, ui_5, 0) as ui_5,
    if(ui_6>0, ui_6, 0) as ui_6,
    if(ui_7>0, ui_7, 0) as ui_7,
    if(ui_8>0, ui_8, 0) as ui_8,
    if(ui_9>0, ui_9, 0) as ui_9,
    if(ui_10>0, ui_9, 0) as ui_10,
    if(ui_11>0, ui_11, 0) as ui_11,
    if(ui_12>0, ui_12, 0) as ui_12,
    if(u_1>0, u_1, 0) as u_1,
    if(u_2>0, u_2, 0) as u_2,
    if(u_3>0, u_3, 0) as u_3,
    if(u_4>0, u_4, 0) as u_4,
    if(u_5>0, u_5, 0) as u_5
from test_label_10 l
left join test_1_item_features_10 f1 on l.item_id=f1.item_id
left join test_2_item_features_10 f2 on l.item_id=f2.item_id
left join test_1_ui_features_10 f3 on l.item_id=f3.item_id and l.user_id=f3.user_id
left join test_2_ui_features_10 f4 on l.item_id=f4.item_id and l.user_id=f4.user_id
left join test_1_user_features_10 f5 on l.user_id=f5.user_id;

insert overwrite local directory '/home/hadoop/yjsdir/DATAPRO/AliRecommendation/data/submit_data/' row format delimited fields terminated by ',' select * from test_10;
--------------------------------------------提交集结束---------------------------------------------------------------
