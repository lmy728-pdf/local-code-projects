# Controlled numerical check ONLY: saved R bandwidths, no bandwidth optimization.
.libPaths(c(normalizePath('R_library'),.Library))
library(np);library(jsonlite)
options(np.messages=FALSE,digits=17)
defaults <- options()[startsWith(names(options()),'np.')]
defaults$actual_nmulti_one_continuous_predictor <- get('npDefaultNmulti',asNamespace('np'))(1L)
write_json(defaults,'np_runtime_options.json',auto_unbox=TRUE,pretty=TRUE)
# np 0.80-1 source R/zzz.R enables this large-bandwidth approximation by default.
# Keep author-default results untouched; disable only this optimization here.
options(np.largeh=FALSE)
dir.create('r_fixed_bandwidth_exact',showWarnings=FALSE)
samples <- read.csv('selected_samples.csv',stringsAsFactors=FALSE)
rows <- list()
for(i in seq_len(nrow(samples))) {
  stem <- samples$stem[i]
  x <- read.csv(file.path('inputs',paste0(stem,'.csv')))
  saved <- readRDS(file.path('r_results',paste0(stem,'_bandwidths.rds')))
  a <- x$D; y <- x$y_transformed; g <- x$g_initial; m <- x$m_initial
  G <- M <- rD <- rY <- u <- rep(NA_real_,nrow(x))
  for(k in 1:5) {
    ix <- x$fold==k
    G[ix] <- fitted(npreg(bws=saved[[k]]$G,txdat=g[ix],tydat=y[ix]-m[ix]))
    M[ix] <- fitted(npreg(bws=saved[[k]]$M,txdat=m[ix],tydat=a[ix]-g[ix]))
    rd <- lm(a[ix]~-1+offset(g[ix])+G[ix])
    ry <- lm(I(y[ix])~-1+offset(m[ix])+M[ix])
    rD[ix] <- a[ix]-rd$fitted; rY[ix] <- y[ix]-ry$fitted
    u[ix] <- rD[ix]*rY[ix]-G[ix]*rD[ix]-M[ix]*rY[ix]
  }
  write.csv(data.frame(observation_id=x$observation_id,fold=x$fold,G_aux=G,M_aux=M,score=u),
            file.path('r_fixed_bandwidth_exact',paste0(stem,'.csv')),row.names=FALSE)
  rows[[i]] <- data.frame(scenario=samples$scenario[i],replication=samples$replication[i],
                          T_R_largeh_off=sqrt(length(u))*mean(u)/sd(u),
                          score_mean=mean(u),score_sd=sd(u),np_largeh=FALSE)
}
write.csv(do.call(rbind,rows),'fixed_bandwidth_numeric_summary.csv',row.names=FALSE)
cat('Fixed-bandwidth numerical checks complete; no bandwidths reselected.\n')
