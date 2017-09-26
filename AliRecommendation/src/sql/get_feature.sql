use tb;

select 
    feature1.item_category,
    case behavior_type
        when 4 then '1'
        else '0'
    end as label
from 
    tianchi_fresh_comp_train_user l
left join (
    select
        item_id,
        item_category
    from 
        tianchi_fresh_comp_train_user
    where
        time>='${hiveconf:f_w_s}' and time<'${hiveconf:l_w_s}'
) feature1 on l.item_id = feature1.item_id
where 
    time>='${hiveconf:l_w_s}' and time<='${hiveconf:l_w_e}'

