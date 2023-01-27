parentdir=$(dirname `pwd`)
SM_DIR=${parentdir}

SMDATA_DIR=${SM_DIR}/data

# opt. activate env, e.g.
# source conda_path
# conda activate env

for COLL in dataset_name1
do
	echo "Extracting json for ${COLL}"
	python extract_json.py \
	${SMDATA_DIR}/raw/${COLL} \
	${SMDATA_DIR}/json_raw/${COLL}
	
	python filter_json.py \
	${SMDATA_DIR}/json_raw/${COLL} \
	${SMDATA_DIR}/json/${COLL} \
	--chart_types=dance-single \
	--chart_difficulties=Beginner,Easy,Medium,Hard,Challenge \
	--min_chart_feet=1 \
	--max_chart_feet=-1 \
	--substitutions=M,0,4,2 \
	--arrow_types=1,2,3 \
	--max_jump_size=-1 \
	--remove_zeros \
	--permutations=0123,3120,0213,3210
	echo "--------------------------------------------"
done