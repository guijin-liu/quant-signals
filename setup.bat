@echo off
chcp 65001 >nul
echo ================================================
echo   量化交易系统 - 环境安装脚本
echo ================================================
echo.

cd /d "C:\Users\Administrator\quant_trading"

echo [1/2] 安装Python依赖包...
echo 这可能需要几分钟，请等待...
echo.

pip install pandas numpy scikit-learn matplotlib joblib pyarrow tqdm akshare yfinance snownlp openpyxl

echo.
echo [2/2] 验证安装...
echo.

python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import akshare; print('akshare:', akshare.__version__)"
python -c "import yfinance; print('yfinance OK')"
python -c "import snownlp; print('snownlp OK')"
python -c "import pyarrow; print('pyarrow OK')"
python -c "import joblib; print('joblib OK')"
python -c "import tqdm; print('tqdm OK')"

echo.
echo ================================================
echo   安装完成！
echo   运行: python main.py all    (完整流程)
echo         python main.py backtest (回测)
echo         python main.py signal   (信号)
echo ================================================
pause
