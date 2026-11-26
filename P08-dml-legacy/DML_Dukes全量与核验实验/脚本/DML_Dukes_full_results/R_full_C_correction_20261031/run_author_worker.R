# Direct calls to the unchanged pinned author function; no diagnostic substitute.
reference <- normalizePath('../R_cross_validation',mustWork=TRUE)
.libPaths(c(file.path(reference,'R_library'),.Library))
suppressPackageStartupMessages(library(np))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(packageVersion('np')==package_version('0.80-1'),getRversion()==package_version('4.6.1'))
options(np.messages=FALSE,digits=17)
RNGkind('Mersenne-Twister','Inversion','Rejection')
source('reference/dri_functions.r',local=.GlobalEnv)
args <- commandArgs(trailingOnly=TRUE)
worker <- as.integer(args[1]);workers <- as.integer(args[2])
stopifnot(worker>=0,worker<workers)
samples <- read.csv('replications.csv',stringsAsFactors=FALSE)
indices <- which((seq_len(nrow(samples))-1L) %% workers == worker)
if(worker==0) {
  capture.output(sessionInfo(),file='r_session_info.txt')
  write_json(list(R_version=R.version.string,np_version=as.character(packageVersion('np')),
                  RNGkind=RNGkind(),np_largeh=getOption('np.largeh'),np_tree=getOption('np.tree'),
                  np_largeh_rel_tol=getOption('np.largeh.rel.tol'),
                  nmulti_one_predictor=get('npDefaultNmulti',asNamespace('np'))(1L)),
             'r_runtime.json',auto_unbox=TRUE,pretty=TRUE)
}
for(i in indices) {
  meta <- samples[i,];path <- file.path('results',paste0(meta$stem,'.json'))
  if(file.exists(path)) {
    previous <- read_json(path,simplifyVector=TRUE)
    stopifnot(previous$input_sha256==meta$input_sha256,previous$r_auxiliary_seed==meta$r_auxiliary_seed)
    next
  }
  started <- proc.time()[['elapsed']];warnings <- character()
  answer <- tryCatch(withCallingHandlers({
    input <- read.csv(file.path('inputs',paste0(meta$stem,'.csv')))
    stopifnot(nrow(input)==1000,identical(input$observation_id,1:1000),
              identical(sort(unique(input$fold)),1:5),all(tabulate(input$fold)==200),
              all(is.finite(as.matrix(input))))
    set.seed(meta$r_auxiliary_seed)
    original <- rdml_cross.test(N=1000,y=input$y_transformed,a=input$D,
                               G=input$g_initial,M=input$m_initial,nsplits=5,s=input$fold)
    if(any(!is.finite(original)) || original[['se']]<=0) stop('Nonfinite author output or nonpositive standard error')
    # Author est=sqrt(N)*mean(score), se=sd(score)/sqrt(N); est/se is NOT T.
    T <- unname(original[['est']]/(original[['se']]*sqrt(1000)))
    list(status='ok',author_original=as.list(original),T_R=T,p_R=unname(original[['p_val']]),
         score_mean=unname(original[['est']]/sqrt(1000)),score_sd=unname(original[['se']]*sqrt(1000)),
         accepted=unname(original[['p_val']]>=0.05),nonfinite_output=FALSE)
  },warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
  error=function(e)list(status='failed',error=conditionMessage(e)))
  answer$scenario <- meta$scenario;answer$replication <- meta$replication;answer$stem <- meta$stem
  answer$r_auxiliary_seed <- meta$r_auxiliary_seed;answer$input_sha256 <- meta$input_sha256
  answer$warnings <- warnings;answer$elapsed_seconds <- proc.time()[['elapsed']]-started
  answer$worker <- worker
  temporary <- paste0(path,'.tmp')
  write_json(answer,temporary,auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
  stopifnot(file.rename(temporary,path))
  cat(meta$stem,answer$status,'seconds',answer$elapsed_seconds,'\n');flush.console()
}
cat('WORKER_COMPLETE',worker,'\n')
