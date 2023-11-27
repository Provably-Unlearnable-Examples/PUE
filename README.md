### The code of the paper "Provably Unlearnable Examples"

**Launch a recovery attack against a classifier trained on PAP noise**

```
python recover.py       --version                 resnet18_recover \
                        --exp_name                ${exp_name} \
                        --config_path             configs/cifar10 \
                        --train_data_type         CIFAR10 \
                        --test_data_type          CIFAR10 \
                        --train_batch_size        128 \
                        --eta                     ${eta} \
                        --surrogate_path          ${surrogate_path} \
                        --load_model 
                        --use_train_subset     # turn it on to use train subset, otherwise use the test set
```


**Crafting PUEs:**

```
python make_pue.py    --config_path          ${config_path} \
                      --train_data_type      ${data_type} \
                      --test_data_type       ${data_type} \
                      --version              ${base_version} \
                      --noise_shape          10 3 32 32 \
                      --epsilon              ${epsilon} \
                      --num_steps            ${num_steps} \
                      --step_size            ${step_size} \
                      --train_step           ${train_step}     \
                      --attack_type          ${attack_type} \
                      --perturb_type         ${perturb_type} \
                      --robust_noise         ${robust_noise}\
                      --avgtimes_perturb     ${u_p}  \
                      --exp_name             ${exp_name} \
                      --universal_stop_error ${universal_stop_error} \
                      --data_parallel \
                      --universal_train_target ${universal_train_target} \
                      --use_subset    
```

**To ```certify``` a surrogate trained on unlearnable examples:**

```
python certify.py --exp_name $exp_name \
                  --config_path $config_path \
                  --surrogate_path $surrogate_path \
                  --train_data_type $data_type \
                  --test_data_type $data_type \
                  --version $base_version \
                  --perturb_type $perturb_type \
                  --sigma $sigma \
                  --q $q \
                  --N $N
```
