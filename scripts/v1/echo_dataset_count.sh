echo "Counting the number of files in each directory"

echo ""

echo "Classify Directory"
echo "+++++++++++++++++++++++++++++++"
echo "Equal Symbol"
find ./workdir/Classify/EqualSymbol -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"
echo "Operator"
find ./workdir/Classify/Operator -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"
echo "Digit"
find ./workdir/Classify/Digit -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"

echo ""

echo "Datasets Directory"
echo "+++++++++++++++++++++++++++++++"
echo "Equal Symbol"
find ./workdir/Datasets/EqualSymbol -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"
echo "Operator"
find ./workdir/Datasets/Operator -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"
echo "Digit"
find ./workdir/Datasets/Digit -type d -exec sh -c 'echo -n "{}: "; ls -1 "{}" | wc -l' \;
echo "+++++++++++++++++++++++++++++++"
