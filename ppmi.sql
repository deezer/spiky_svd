/* DATA PREPARATION
   input: a binary matrix M `input_table` between some 'items' and some 'contexts' (e.g., track and playlists)
     1) transforms as MM^*
     2) filters small values and top-k cooccurences of popular items
     3) applies a PPMI conversion
   returns: a positive symmetric matrix */

with input as (select tid as track_id, pid as context_id from input_table),
  count_t1 as (select context_id, track_id as t1 from input),
  count_t2 as (select context_id, track_id as t2 from input),
  sum_t as (select t1 as t, count(*) as tot from count_t1 group by t1),
  count_t12 as (select t1, t2 from count_t2 left join count_t1 on count_t1.context_id = count_t2.context_id where t1 < t2),
  cooc as (select t1, t2, count(*) as c from count_t12 group by t1, t2),
  ranked as (select t1, t2, c, row_number() over (partition by t1 order by c desc) as rank1, row_number() over (partition by t2 order by c desc) as rank2 from cooc),
  filtered as (select t1, t2, c, rank1, rank2 from ranked where c >= 2  and rank1 < 2000 and rank2 < 2000),
  aug1 as (select t1, t2, tot as c1, c from filtered inner join sum_t on t1 = t),
  aug2 as (select t1, t2, c1, tot as c2, c from aug1 inner join sum_t on t2 = t),
  final_matrix as (select t1, t2, LOG(@CST * c / c1 / c2) as pmi from aug2)
  /* we replace @CST by the total number of contexts, also computable as
   `select count(distinct context_id) from input_table` */
   
select t1, t2, pmi from final_matrix where pmi > 0