use faer::prelude::Solve;
use faer::Mat;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[derive(Deserialize)]
struct DimsonKwargs {
    min_obs: usize,
}

/// Sum of market-return coefficients from a Dimson regression.
///
/// `inputs[0]` is the target series (`ret_exc`); `inputs[1..]` are the market
/// regressors (`mktrf`, `mktrf_ld1`, `mktrf_lg1`, ...). An intercept is
/// appended automatically. The output is a single-element `Float64` Series
/// containing the sum of all market-return coefficients (intercept excluded),
/// or null if the group is too small or X'X is singular.
#[polars_expr(output_type = Float64)]
fn dimson_beta(inputs: &[Series], kwargs: DimsonKwargs) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        return Err(PolarsError::ComputeError(
            "dimson_beta requires at least one market regressor".into(),
        ));
    }

    let n = inputs[0].len();
    let n_mkt = inputs.len() - 1;
    let p = n_mkt + 1; // +1 for intercept

    let null_out = || Series::new("dimson_beta".into(), &[None::<f64>]);

    if n < kwargs.min_obs.max(p + 1) {
        return Ok(null_out());
    }

    // Zero-copy slice access. base_data_filter_exp drops nulls upstream and
    // group_by aggregation produces a single contiguous chunk per group.
    let target_ca = inputs[0].f64()?;
    let target = match target_ca.cont_slice() {
        Ok(s) => s,
        Err(_) => return Ok(null_out()),
    };

    let mut mkt: Vec<&[f64]> = Vec::with_capacity(n_mkt);
    for s in &inputs[1..] {
        let ca = s.f64()?;
        match ca.cont_slice() {
            Ok(slice) => mkt.push(slice),
            Err(_) => return Ok(null_out()),
        }
    }

    // X'X is symmetric (p × p). Build only the lower triangle and mirror.
    // X column j (j < n_mkt) is mkt[j]; column n_mkt is the intercept (1.0).
    // X'y is length p.
    let mut xtx = vec![0.0_f64; p * p];
    let mut xty = vec![0.0_f64; p];

    let n_f = n as f64;

    // Cross-products among market columns.
    for i in 0..n_mkt {
        let xi = mkt[i];
        // diagonal
        let mut dot = 0.0;
        let mut sum_i = 0.0;
        let mut xy = 0.0;
        for k in 0..n {
            let v = xi[k];
            dot += v * v;
            sum_i += v;
            xy += v * target[k];
        }
        xtx[i * p + i] = dot;
        xtx[n_mkt * p + i] = sum_i; // intercept row × col i (sum of x_i)
        xtx[i * p + n_mkt] = sum_i; // mirror
        xty[i] = xy;

        // off-diagonal market × market
        for j in (i + 1)..n_mkt {
            let xj = mkt[j];
            let mut cross = 0.0;
            for k in 0..n {
                cross += xi[k] * xj[k];
            }
            xtx[i * p + j] = cross;
            xtx[j * p + i] = cross;
        }
    }
    // Intercept row/col diagonal and X'y last entry.
    xtx[n_mkt * p + n_mkt] = n_f;
    let mut sum_y = 0.0;
    for k in 0..n {
        sum_y += target[k];
    }
    xty[n_mkt] = sum_y;

    // Solve (X'X) β = X'y via faer Cholesky.
    let xtx_mat = Mat::<f64>::from_fn(p, p, |i, j| xtx[i * p + j]);
    let xty_mat = Mat::<f64>::from_fn(p, 1, |i, _| xty[i]);

    let llt = match xtx_mat.llt(faer::Side::Lower) {
        Ok(decomp) => decomp,
        Err(_) => return Ok(null_out()),
    };
    let beta = llt.solve(&xty_mat);

    let mut sum_mkt: f64 = 0.0;
    for i in 0..n_mkt {
        sum_mkt += *beta.get(i, 0);
    }

    if !sum_mkt.is_finite() {
        return Ok(null_out());
    }

    Ok(Series::new("dimson_beta".into(), &[Some(sum_mkt)]))
}
