#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass(frozen=True)
class Flower:
    name: str
    price: float


class FlowerShopApp:
    def __init__(self) -> None:
        self.flowers = {
            "1": Flower("Роза", 150.0),
            "2": Flower("Тюльпан", 90.0),
            "3": Flower("Лилия", 170.0),
            "4": Flower("Пион", 210.0),
        }
        self.cart: dict[str, int] = {}

    def show_flowers(self) -> None:
        print("\nКаталог цветов:")
        for code, flower in self.flowers.items():
            print(f"{code}. {flower.name} — {flower.price:.2f} ₽")

    def add_to_cart(self) -> None:
        self.show_flowers()
        code = input("Введите номер цветка: ").strip()
        if code not in self.flowers:
            print("Такого номера нет в каталоге.")
            return
        quantity_raw = input("Введите количество: ").strip()
        try:
            quantity = int(quantity_raw)
        except ValueError:
            print("Количество должно быть целым числом.")
            return
        if quantity <= 0:
            print("Количество должно быть больше нуля.")
            return
        self.cart[code] = self.cart.get(code, 0) + quantity
        print("Товар добавлен в корзину.")

    def remove_from_cart(self) -> None:
        if not self.cart:
            print("Корзина пуста.")
            return
        self.show_cart()
        code = input("Введите номер цветка для удаления: ").strip()
        if code not in self.cart:
            print("Такого товара нет в корзине.")
            return
        del self.cart[code]
        print("Товар удален из корзины.")

    def show_cart(self) -> None:
        if not self.cart:
            print("\nКорзина пуста.")
            return
        print("\nКорзина:")
        total = 0.0
        for code, quantity in self.cart.items():
            flower = self.flowers[code]
            item_total = flower.price * quantity
            total += item_total
            print(f"- {flower.name}: {quantity} шт. x {flower.price:.2f} ₽ = {item_total:.2f} ₽")
        print(f"Итого: {total:.2f} ₽")

    def checkout(self) -> None:
        if not self.cart:
            print("Нельзя оформить заказ: корзина пуста.")
            return
        self.show_cart()
        print("Заказ оформлен. Спасибо за покупку!")
        self.cart.clear()

    def run(self) -> None:
        while True:
            print(
                "\n=== Магазин цветов ===\n"
                "1. Показать каталог\n"
                "2. Добавить в корзину\n"
                "3. Удалить из корзины\n"
                "4. Показать корзину\n"
                "5. Оформить заказ\n"
                "0. Выход"
            )
            choice = input("Выберите действие: ").strip()
            if choice == "1":
                self.show_flowers()
            elif choice == "2":
                self.add_to_cart()
            elif choice == "3":
                self.remove_from_cart()
            elif choice == "4":
                self.show_cart()
            elif choice == "5":
                self.checkout()
            elif choice == "0":
                print("До свидания!")
                break
            else:
                print("Неизвестная команда.")


if __name__ == "__main__":
    FlowerShopApp().run()
