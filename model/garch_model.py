
from arch import arch_model
import scipy.stats as stats

def forecast_garch_var(df, alpha=0.05, horizon=5):
    returns = df['Returns'] * 100
    model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=horizon)
    vol_forecast = garch_forecast.variance.iloc[-1] ** 0.5
    z_score = stats.norm.ppf(alpha)
    var_1d = -(z_score * vol_forecast[0])
    return vol_forecast, var_1d
