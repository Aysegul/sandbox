#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "LeakyReLU.cu"


LUA_EXTERNC DLL_EXPORT int luaopen_libacunn(lua_State *L);

int luaopen_libacunn(lua_State *L)
{
  lua_newtable(L);

  cunn_LeakyReLU_init(L);

  return 1;
}
