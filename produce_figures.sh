echo "Running label_accuracy_simulations.py for all distributions"
echo "python label_accuracy_simulations.py --distribution <distribution> --mehtod <method>"

# Provide all arguments: --distribution, --method, and --metric
python label_accuracy_simulations.py --distribution uniform --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution uniform_10 --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution uniform_30 --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution uniform_50 --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution normal_variance --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution normal_mean --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution dog_and_baby --method FIX --n_samples $1
python label_accuracy_simulations.py --distribution gamma --method FIX --n_samples $1

echo "Running annotation_cost.py"
python cost_simulations.py
