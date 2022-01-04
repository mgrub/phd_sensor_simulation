cd C:\Users\gruber04\git_repos\PhD\phd_sensor_simulation

rem copy pictures and result files
scp i01155:/home/gruber04/git_repos/phd_sensor_simulation/results/*.png .\results\
scp i01155:/home/gruber04/git_repos/phd_sensor_simulation/results/*.json .\results\

rem move file to other directory on host
ssh i01155 "mv /home/gruber04/git_repos/phd_sensor_simulation/results/*.{png,json} /home/gruber04/git_repos/phd_sensor_simulation/results/already_copied/"
