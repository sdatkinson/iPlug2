#!/bin/sh

cd "$(dirname "$0")"

cd ..

pwd

# trash the whole build-web folder
if [ -d build-web ]
then
  rm -r build-web
fi

mkdir build-web

# trash only the scripts folder - uncomment if you have customised your html
# if [ -d build-web/scripts ]
# then
#   rm -r build-web/scripts
# fi

mkdir build-web/scripts

echo BUNDLING RESOURCES -----------------------------
python /Users/oli/Dev/MyWeb/emscripten1.37.22/emscripten/1.37.22/tools/file_packager.py ./build-web/resources.data --use-preload-plugins --preload ./resources/img@/ --js-output=./build-web/resources.js
echo -

echo MAKING  - WAM WASM MODULE -----------------------------
emmake make --makefile projects/IPlugEffect-wam.mk

if ! [ -a build-web/scripts/IPlugEffect-WAM.wasm ]
then
  echo WAM compilation failed
  exit
fi

cd build-web/scripts
# thanks to Steven Yi / Csound
echo "
fs = require('fs');
let wasmData = fs.readFileSync(\"IPlugEffect-WAM.wasm\");
let wasmStr = wasmData.join(\",\");
let wasmOut = \"AudioWorkletGlobalScope.WAM = AudioWorkletGlobalScope.WAM || {}\\\n\";
wasmOut += \"AudioWorkletGlobalScope.WAM.IPlug = { ENVIRONMENT: 'WEB' }\\\n\";
wasmOut += \"AudioWorkletGlobalScope.WAM.IPlug.wasmBinary = new Uint8Array([\" + wasmStr + \"]);\";
fs.writeFileSync(\"IPlugEffect-WAM.wasm.js\", wasmOut);
// later we will possibly switch to base64
// as suggested by Stephane Letz / Faust
// let base64 = wasmData.toString(\"base64\");
// fs.writeFileSync(\"IPlugEffect-WAM.base64.js\", base64);
" > encode-wasm.js

node encode-wasm.js
rm encode-wasm.js

cp ../../../../Dependencies/IPlug/WAM_SDK/wamsdk/*.js .
cp ../../../../Dependencies/IPlug/WAM_AWP/*.js .
cp ../../../../IPlug/WEB/Template/scripts/IPlugWAM-awn.js IPlugEffect-awn.js
sed -i "" s/IPlugWAM/IPlugEffect/g IPlugEffect-awn.js
cp ../../../../IPlug/WEB/Template/scripts/IPlugWAM-awp.js IPlugEffect-awp.js
sed -i "" s/IPlugWAM/IPlugEffect/g IPlugEffect-awp.js
cd ..

#copy in the template html - comment if you have customised the html
cp ../../../IPlug/WEB/Template/IPlugWAM-standalone.html index.html
sed -i "" s/IPlugWAM/IPlugEffect/g index.html

cd ../

echo
echo MAKING  - WEB WASM MODULE -----------------------------

emmake make --makefile projects/IPlugEffect-web.mk

mv build-web/scripts/*.wasm build-web

if [ -a build-web/IPlugEffect.wasm ]
then
  cd build-web
  emrun --no_emrun_detect --browser chrome index.html
  #   emrun --browser firefox index.html
fi
