input_file=$1

# mkdir review_data

while read line; do 
	url="$(echo $line | cut -d' ' -f1)"
	
	output_file="$(echo $line |cut -d' ' -f2)"
	
	curl "${url}" > "review_data/${output_file}.txt"
	curl "${url}?start=20" >> "review_data/${output_file}.txt"
	curl "${url}?start=40" >> "review_data/${output_file}.txt"
	curl "${url}?start=60" >> "review_data/${output_file}.txt"
	curl "${url}?start=80" >> "review_data/${output_file}.txt"
	
done < $input_file