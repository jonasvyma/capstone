create table startwars_mocloose_sales as
with moc as(
select
	figure,
	authenticity_n,
	 ROUND(AVG(selling_price)::numeric, 2) as selling_price, 
	 count(selling_price) as sales,
	"condition", 
	character_type,
	year::text as year,
	"month" 
from
	the_collectors.starwars_mocfigures
group by
	1,
	2,
	5,
	6, 7, 8
),
loose as (
select
	figure,
	authenticity_n,
	selling_price_usd as selling_price,
	recorded_sales_updated as sales,
	"condition",
	character_type,
	year::text as year
from
	the_collectors.starwars_loosefigures
)
select
	figure,
	authenticity_n,
	selling_price,
	sales,
	"condition",
	character_type,
	year
from
	moc
union all
select
	*
from
	loose
order by
	figure,
	character_type,
	condition,
	authenticity_n,
	year asc;
