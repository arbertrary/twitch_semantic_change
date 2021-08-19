printf "DUREL:\n"

sg1dir="/home/stud/bernstetter/models/woc_reproduce/results/durel_sg1/"
sg0dir="/home/stud/bernstetter/models/woc_reproduce/results/durel_sg0/"
targets="/home/stud/bernstetter/ma/LSCDetection/testsets/durel/rank.tsv"
cos="cosine.tsv"
neigh="neighborhood.tsv"
WOC_DIR="..."

echo $sg1dir
echo $sg0dir
printf "\n"

python3 $WOC_DIR/evaluation/spr.py $targets "$sg0dir/$cos" "durel targets" "sg0cos"  0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg0dir/$neigh" "durel targets" "sg0neigh" 0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg1dir/$cos" "durel targets" "sg1cos" 0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg1dir/$neigh" "durel targets" "sg1neigh" 0 2
printf "##### \n"

printf "SUREL:\n"

sg1dir="/home/stud/bernstetter/models/woc_reproduce/results/surel_sg1/"
sg0dir="/home/stud/bernstetter/models/woc_reproduce/results/surel_sg0/"
targets="/home/stud/bernstetter/ma/LSCDetection/testsets/surel/rank.tsv"

python3 $WOC_DIR/evaluation/spr.py $targets "$sg0dir/$cos" "surel targets" "sg0cos"  0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg0dir/$neigh" "surel targets" "sg0neigh" 0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg1dir/$cos" "surel targets" "sg1cos" 0 2
printf "\n"
python3 $WOC_DIR/evaluation/spr.py $targets "$sg1dir/$neigh" "surel targets" "sg1neigh" 0 2
printf "##### \n"
