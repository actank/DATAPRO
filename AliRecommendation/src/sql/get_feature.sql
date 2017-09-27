use tb;

--set hive.cli.print.header=true;
drop table if exists ${hiveconf:train_table};
--drop table if exists train_data_10;
create table ${hiveconf:train_table} as
select
item_id as item_id, 
item_category as item_category, 
if(favorite_num_10>0, favorite_num_10, 0) as favorite_num_10,
if(cart_num_10>0, cart_num_10, 0) as cart_num_10,
if(click_num_10>0, click_num_10, 0) as click_num_10,
label
from(
select
    l.item_id as item_id,
    l.item_category as item_category,
    feature1.favorite_num_10 as favorite_num_10,
    feature2.cart_num_10 as cart_num_10,
    feature3.click_num_10 as click_num_10,
    case behavior_type
        when 4 then '1'
        else '0'
    end as label
from
    tianchi_fresh_comp_train_user l
--十天收藏数
left join (
    select
        item_id,
        count(item_id) as favorite_num_10
    from
        tianchi_fresh_comp_train_user
    where
        time>='${hiveconf:f_w_s}' and time<'${hiveconf:l_w_s}' and behavior_type=2
    group by item_id
) feature1 on l.item_id = feature1.item_id
left join(
--十天加车数
select
        item_id,
        count(item_id) as cart_num_10
    from
        tianchi_fresh_comp_train_user
    where
        time>='${hiveconf:f_w_s}' and time<'${hiveconf:l_w_s}' and behavior_type=3
    group by item_id
) feature2 on l.item_id = feature2.item_id
left join(
--十天浏览数
select
        item_id,
        count(item_id) as click_num_10
    from
        tianchi_fresh_comp_train_user
    where
        time>='${hiveconf:f_w_s}' and time<'${hiveconf:l_w_s}' and behavior_type=1
    group by item_id
) feature3 on l.item_id = feature3.item_id
where
    time>='${hiveconf:l_w_s}' and time<='${hiveconf:l_w_e}' and l.item_id is not null 
) tt
group by item_id, item_category, favorite_num_10, cart_num_10, click_num_10, label
