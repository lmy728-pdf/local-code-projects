.libPaths(c(normalizePath('R_library'), .Library))
library(np)
capture.output(sessionInfo(),file='r_session_info.txt')
capture.output(packageDescription('np'),file='np_package_description.txt')
sink('np_installed_defaults.txt')
for (name in c('npregbw.default','npregbw.rbandwidth','npreg.default','npreg.rbandwidth')) {
  cat('\nFUNCTION ',name,'\n')
  fun=get(name,envir=asNamespace('np'))
  print(formals(fun)); print(body(fun))
}
sink()
cat('np version:',as.character(packageVersion('np')),'\n')
