FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["f_ml_test_core/f_ml_test_core.fsproj", "f_ml_test_core/"]
RUN dotnet restore "f_ml_test_core/f_ml_test_core.fsproj"
COPY . .
WORKDIR "/src/f_ml_test_core"
RUN dotnet build "f_ml_test_core.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "f_ml_test_core.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "f_ml_test_core.dll"]
