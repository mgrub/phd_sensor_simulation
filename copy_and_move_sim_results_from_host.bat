rem copy pictures to desktop
scp i01155:/home/gruber04/git_repos/phd_sensor_simulation/results/*.{png,json} .\Desktop\results_sim\

rem move file to other directory on host
ssh i01155 "mv /home/gruber04/git_repos/phd_sensor_simulation/results/*.{png,json} /home/gruber04/git_repos/phd_sensor_simulation/results/already_copied/"
