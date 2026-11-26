# Uses the pinned author file unmodified; diagnostic wrapper follows its order.
.libPaths(c(normalizePath('R_library'), .Library))
library(np)
library(jsonlite)
options(np.messages=FALSE, digits=17)
Sys.setenv(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
RNGkind('Mersenne-Twister','Inversion','Rejection')
source('downloads/dri_functions.r',local=.GlobalEnv)
dir.create('r_results',showWarnings=FALSE)
capture.output(sessionInfo(),file='r_session_info.txt')
capture.output(formals(get('rbandwidth',envir=asNamespace('np'))),file='rbandwidth_defaults.txt')

diagnostic_wrapper <- function(N,y,a,G,M,nsplits,s) {
  V <- Z <- Ms <- Gs <- Gnew <- Mnew <- u <- rep(NA_real_,N)
  details <- list(); bandwidth_objects <- list()
  scalar <- function(object,name) {
    value <- object[[name]]
    if(is.null(value)) return(NA_character_)
    paste(as.character(value),collapse=';')
  }
  for(vfold in 1:nsplits) {
    train <- s!=vfold; test <- s==vfold
    if(nsplits==1) train <- test
    V[test] <- a[test]-G[test]
    if(length(unique(M[test]))==1) {
      Gs[test] <- predict(lm(V[test]~-1+(M[test])))
      bw_G <- NULL
    } else {
      bw_G <- npregbw(formula=V[test]~M[test])
      Gs[test] <- fitted(npreg(bws=bw_G))
    }
    Z[test] <- y[test]-M[test]
    if(length(unique(G[test]))==1) {
      Ms[test] <- predict(lm(Z[test]~-1+(G[test])))
      bw_M <- NULL
    } else {
      bw_M <- npregbw(formula=Z[test]~G[test])
      Ms[test] <- fitted(npreg(bws=bw_M))
    }
    fit_M <- lm(I(y[test])~-1+offset(M[test])+Gs[test])
    fit_G <- lm(a[test]~-1+offset(G[test])+Ms[test])
    Mnew[test] <- fit_M$fitted
    Gnew[test] <- fit_G$fitted
    u[test] <- (a[test]-Gnew[test])*(y[test]-Mnew[test])-Gs[test]*(y[test]-Mnew[test])-Ms[test]*(a[test]-Gnew[test])
    # Translate author naming to the Python/paper G and M convention.
    row <- list(fold=vfold,bandwidth_G=if(is.null(bw_M)) NA_real_ else as.numeric(bw_M$bw),
                bandwidth_M=if(is.null(bw_G)) NA_real_ else as.numeric(bw_G$bw),
                alpha=unname(coef(fit_G)[1]),beta=unname(coef(fit_M)[1]),
                cv_loss_G=if(is.null(bw_M)) NA_real_ else bw_M$fval,
                cv_loss_M=if(is.null(bw_G)) NA_real_ else bw_G$fval,
                constant_G=is.null(bw_M),constant_M=is.null(bw_G),
                mean_G_rD=mean(Ms[test]*(a[test]-Gnew[test])),
                mean_M_rY=mean(Gs[test]*(y[test]-Mnew[test])))
    for(name in c('regtype','method','type','ckertype','ckerorder','scaling','nmulti','ckerbound','ncon','itmax','ftol','tol')) {
      row[[paste0(name,'_G')]] <- scalar(bw_M,name)
      row[[paste0(name,'_M')]] <- scalar(bw_G,name)
    }
    details[[vfold]] <- as.data.frame(row,stringsAsFactors=FALSE)
    bandwidth_objects[[vfold]] <- list(G=bw_M,M=bw_G)
  }
  variance <- var(u)
  answer <- c(est=mean(u)*sqrt(N),se=sqrt(variance/N),p_val=pchisq(mean(u)^2/(variance/N),1,lower.tail=FALSE))
  list(answer=answer,folds=do.call(rbind,details),bandwidth_objects=bandwidth_objects,
       arrays=data.frame(observation_id=1:N,fold=s,G_aux=Ms,M_aux=Gs,Gnew=Gnew,Mnew=Mnew,
                         rD=a-Gnew,rY=y-Mnew,score=u),
       summary=list(mean_score=mean(u),sd_score=sd(u),variance=variance,T=sqrt(N)*mean(u)/sd(u),p_value=answer[['p_val']]))
}

samples <- read.csv('selected_samples.csv',stringsAsFactors=FALSE)
args <- commandArgs(trailingOnly=TRUE)
force <- '--force' %in% args
limits <- args[args!='--force']
if(length(limits)) samples <- samples[seq_len(as.integer(limits[1])),,drop=FALSE]
for(i in seq_len(nrow(samples))) {
  meta <- samples[i,]; stem <- meta$stem
  marker <- file.path('r_results',paste0(stem,'.json'))
  if(file.exists(marker) && !force) {cat('Existing ',stem,' skipped\n');next}
  input <- read.csv(file.path('inputs',paste0(stem,'.csv')),check.names=FALSE)
  stopifnot(nrow(input)==1000,identical(input$observation_id,1:1000),all(tabulate(input$fold)==200))
  N <- nrow(input); y <- input$y_transformed; a <- input$D; G <- input$g_initial; M <- input$m_initial; s <- input$fold
  started <- proc.time()[['elapsed']]; warns <- character()
  result <- tryCatch(withCallingHandlers({
    set.seed(meta$r_auxiliary_seed)
    original <- rdml_cross.test(N=N,y=y,a=a,G=G,M=M,nsplits=5,s=s)
    original_rng_end <- .Random.seed
    set.seed(meta$r_auxiliary_seed)
    diagnostic <- diagnostic_wrapper(N=N,y=y,a=a,G=G,M=M,nsplits=5,s=s)
    equality <- isTRUE(all.equal(unname(original),unname(diagnostic$answer),tolerance=1e-12))
    rng_equal <- identical(original_rng_end,.Random.seed)
    if(!equality || !rng_equal) stop('Diagnostic wrapper differs from unmodified author output or RNG consumption')
    if(any(!is.finite(as.matrix(diagnostic$arrays)))) stop('Non-finite intermediate R output')
    write.csv(diagnostic$arrays,file.path('r_results',paste0(stem,'_arrays.csv')),row.names=FALSE)
    write.csv(diagnostic$folds,file.path('r_results',paste0(stem,'_folds.csv')),row.names=FALSE)
    saveRDS(diagnostic$bandwidth_objects,file.path('r_results',paste0(stem,'_bandwidths.rds')))
    list(status='ok',scenario=meta$scenario,replication=meta$replication,r_auxiliary_seed=meta$r_auxiliary_seed,
         author_original=as.list(original),wrapper=as.list(diagnostic$answer),wrapper_matches_original=equality,
         rng_consumption_matches=rng_equal,wrapper_max_difference=max(abs(original-diagnostic$answer)),
         # Author 'est' is sqrt(N)*mean(score), NOT a theta estimate or a T.
         T_original=unname(original[['est']]/(original[['se']]*sqrt(N))),
         diagnostics=diagnostic$summary)
  },warning=function(w){warns <<- c(warns,conditionMessage(w));invokeRestart('muffleWarning')}),
  error=function(e) list(status='failed',scenario=meta$scenario,replication=meta$replication,error=conditionMessage(e)))
  result$warnings <- warns; result$elapsed_seconds <- proc.time()[['elapsed']]-started
  result$R_version <- R.version.string; result$np_version <- as.character(packageVersion('np'))
  result$RNGkind <- RNGkind()
  write_json(result,marker,auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
  cat(i,'/',nrow(samples),stem,result$status,'seconds=',result$elapsed_seconds,'\n');flush.console()
}
