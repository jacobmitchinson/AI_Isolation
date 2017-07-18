for i in {1..100}
do
    python tournament.py | tee -a results.txt
done
