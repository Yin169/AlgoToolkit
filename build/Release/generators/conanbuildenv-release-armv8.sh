script_folder="/Users/yincheangng/worksapce/Github/MyAlgorithmicToolkit/build/Release/generators"
echo "echo Restoring environment" > "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
for v in OpenBLAS_HOME
do
    is_defined="true"
    value=$(printenv $v) || is_defined="" || true
    if [ -n "$value" ] || [ -n "$is_defined" ]
    then
        echo export "$v='$value'" >> "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
    else
        echo unset $v >> "$script_folder/deactivate_conanbuildenv-release-armv8.sh"
    fi
done


export OpenBLAS_HOME="/Users/yincheangng/.conan2/p/b/openbfc9fc49749cd8/p"