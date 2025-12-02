# Лабораторная 3
**Гибридная защита: симметричное шифрование + стеганография в изображениях**

## Инструкция по запуску
Для запуска встраивания/восстановления сообщения необходимо создать конфигурацию запуска.

В разделе **script** нужно указать модуль, обрабатывающий входные команды и запускающий встраивание/восстановление сообщения - hybrid_stego.py. В разделе **script parametrs** нужно ввести саму команду (они указаны ниже). В разделе **Working directory** необходимо указать рабочую директорию, это, непосредственно, сама папка **Hybrid**.

## Примеры команд запуска
Команды приведены для картинки gradient.png, но при желании все пути и файлы в командах можно спокойно менять.
> Значение **payload** в конце команды также можно поменять.
### Встраивание сообщения простым LSB-1
```
embed --cover imgs/original/gradient.png --out results/gradient_stego_simple.png --msg message.txt --simple --payload 0.005 --metrics
```
### Восстановление сообщения, встроенного простым LSB-1
```
extract --stego results/gradient_stego_simple.png --out gradient_simple_restored.txt --simple
```
### Встраивание сообщения гибридным LSB-1
```
embed --cover imgs/original/gradient.png --out results/gradient_stego_hybrid.png --msg message.txt --password "mypass" --payload 0.005 --metrics
```
#### Восстановление сообщения, встроенного гибридным LSB
```
extract --stego results/gradient_stego_hybrid.png --out gradient_hybrid_restored.txt --password "mypass"
```

## Docker
Для данного программного модуля был также создан dockerfile (лежит в корне проекта), при помощи которого можно собрать образ и запустить приложение в контейнере.
### Команда сборки образа

